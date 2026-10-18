"""Local JEV System 1 API. Run with the adjacent `serve` launcher."""
import asyncio
import base64
import binascii
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
import io
import json
import logging
import math
import os
import time
from typing import Any, Literal

from paths import HERE, INSTALL, MODEL
os.environ.setdefault("HF_HOME", str(INSTALL / "cache/huggingface"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import mlx.core as mx
from mlx_vlm import load, predict
from PIL import Image, UnidentifiedImageError
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, model_validator
from starlette.middleware.trustedhost import TrustedHostMiddleware

MODEL_ID = "nativ-community/JEV-27B-VL-MLX-8bit"
MAX_BODY_BYTES = 16 * 1024 * 1024
MAX_IMAGE_BYTES = 8 * 1024 * 1024
MAX_INPUT_TOKENS = 16384
MAX_PENDING = 4


class DecideRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["noul", "bool", "choice", "score"]
    state: Any = ""
    question: str = Field(min_length=1, max_length=8000)
    options: list[str] | None = None
    images: list[str] = Field(default_factory=list, max_length=8, description="Base64 data:image/... URLs")
    max_image_dimension: int = Field(default=1024, ge=128, le=2048)

    @model_validator(mode="after")
    def validate_options(self):
        if self.kind == "choice":
            if not self.options or not 2 <= len(self.options) <= 256:
                raise ValueError("choice requires 2 to 256 options")
            if len(set(self.options)) != len(self.options) or any(not x.strip() or len(x) > 4000 for x in self.options):
                raise ValueError("options must be unique, nonempty strings of at most 4000 characters")
        elif self.kind == "score":
            if self.options is not None and self.options != [str(i) for i in range(6)]:
                raise ValueError("score uses levels 0 to 5; describe the rubric in question")
        elif self.options is not None:
            raise ValueError("Boolean decisions do not take options")
        return self


class BodyLimitMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["method"] not in ("POST", "PUT", "PATCH"):
            return await self.app(scope, receive, send)
        chunks, total = [], 0
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            chunk = message.get("body", b"")
            total += len(chunk)
            if total > MAX_BODY_BYTES:
                return await JSONResponse({"detail": "Request exceeds 16 MiB"}, status_code=413)(scope, receive, send)
            chunks.append(chunk)
            if not message.get("more_body", False):
                break
        body, replayed = b"".join(chunks), False

        async def replay():
            nonlocal replayed
            if not replayed:
                replayed = True
                return {"type": "http.request", "body": body, "more_body": False}
            return await receive()

        await self.app(scope, replay, send)


class Engine:
    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="jev-metal")
        self.pending = 0
        self.ready = False
        self.started = time.time()

    def initialize(self):
        if not mx.metal.is_available():
            raise RuntimeError("MLX Metal GPU is unavailable")
        start = time.perf_counter()
        self.model, self.processor = load(str(MODEL), lazy=False)
        mx.synchronize()
        self.load_seconds = time.perf_counter() - start
        self.tokenizer = getattr(self.processor, "tokenizer", self.processor)
        predict(self.model, self.processor, "A customer requests a refund.", {"answer": {"type": "bool", "instructions": "Does the customer request a refund?"}})
        mx.synchronize()
        self.model_memory_bytes = mx.get_active_memory()
        self.device = mx.device_info()["device_name"]
        self.revision = json.loads((HERE / "model-manifest.json").read_text())["revision"]
        self.ready = True
        logging.getLogger("uvicorn.error").info("JEV ready on %s; model memory %.2f GB", self.device, self.model_memory_bytes / 1e9)

    def normalize_state(self, request):
        parts, image_details, images = [], [], []

        def add_image(value):
            if not isinstance(value, str) or not value.startswith("data:image/") or ";base64," not in value:
                raise ValueError("Images must be base64 data:image/... URLs; remote URLs and file paths are not accepted")
            if len(images) >= 8:
                raise ValueError("At most 8 images are accepted")
            encoded = value.split(";base64,", 1)[1]
            if len(encoded) > (MAX_IMAGE_BYTES + 2) // 3 * 4:
                raise ValueError("Each image must be at most 8 MiB")
            try:
                raw = base64.b64decode(encoded, validate=True)
                with Image.open(io.BytesIO(raw)) as source:
                    if source.width * source.height > 20_000_000:
                        raise ValueError("Each image must be at most 20 megapixels")
                    if source.format not in ("PNG", "JPEG", "WEBP"):
                        raise ValueError("Supported image formats are PNG, JPEG, and WebP")
                    original = list(source.size)
                    image = source.convert("RGB")
                image.thumbnail((request.max_image_dimension, request.max_image_dimension), Image.Resampling.LANCZOS)
            except (binascii.Error, UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
                raise ValueError("Invalid or unsupported image data") from exc
            images.append(image)
            parts.append(image)
            image_details.append({"original_size": original, "processed_size": list(image.size)})

        values = request.state if isinstance(request.state, list) else ([] if request.state is None else [request.state if isinstance(request.state, str) else json.dumps(request.state, ensure_ascii=False)])
        for part in values:
            if isinstance(part, dict) and "image" in part:
                add_image(part["image"])
            elif isinstance(part, dict) and part.get("type") == "image_url":
                image_url = part.get("image_url")
                if not isinstance(image_url, dict):
                    raise ValueError("image_url must contain a url field")
                add_image(image_url.get("url"))
            elif isinstance(part, dict) and part.get("type") == "text":
                if not isinstance(part.get("text"), str):
                    raise ValueError("Text parts need a string text field")
                parts.append(part["text"])
            elif part is not None:
                parts.append(part if isinstance(part, str) else json.dumps(part, ensure_ascii=False))
        for value in request.images:
            add_image(value)
        text = "".join(x for x in parts if isinstance(x, str))
        if len(text) > 100000:
            raise ValueError("State text exceeds 100,000 characters")
        # Conservative budget before GPU work. Qwen's spatial merge produces
        # roughly one image token per 32x32 pixels; reserve framing/rounding room.
        text_tokens = len(self.tokenizer.encode(text + request.question + json.dumps(request.options), add_special_tokens=False))
        image_tokens = sum(math.ceil(x.width / 32) * math.ceil(x.height / 32) + 256 for x in images)
        if text_tokens + image_tokens + 512 > MAX_INPUT_TOKENS:
            raise ValueError(f"Request exceeds this local server's conservative {MAX_INPUT_TOKENS}-token budget")
        return parts if images else text, image_details

    def decide(self, request, enqueued_at):
        started = time.perf_counter()
        state, image_details = self.normalize_state(request)
        effective = "noul" if request.kind == "bool" else request.kind
        options = request.options if effective == "choice" else (["false", "true"] if effective == "noul" else [str(i) for i in range(6)])
        question = {"type": "bool" if effective == "noul" else effective, "instructions": request.question}
        if effective in ("choice", "score"):
            question["criteria"] = options
        result = predict(self.model, self.processor, state, {"answer": question})
        mx.synchronize()
        answer = result["answers"]["answer"]
        if effective == "noul":
            p = answer["probability"]
            probabilities = [round(1 - p, 4), p]
        else:
            probabilities = [answer["probabilities"][x] for x in options]
            probability_sum = sum(probabilities)
            probabilities = [x / probability_sum for x in probabilities]
        best = int(answer["value"]) if effective == "noul" else max(range(len(probabilities)), key=probabilities.__getitem__)
        total = result["usage"]["input_tokens"]
        finished = time.perf_counter()
        response = {
            "kind": request.kind, "effective_kind": effective,
            "options": options, "probabilities": probabilities,
            "choice_index": best, "choice": options[best], "confidence": probabilities[best],
            "adaptation": "single" if effective == "choice" and len(options) > 16 else "native",
            "protocol": "jev-mlx-system1", "model": MODEL_ID,
            "usage": {"prompt_tokens": total, "completion_tokens": 0, "total_tokens": total},
            "elapsed_seconds": finished - enqueued_at,
            "inference_seconds": finished - started,
            "queue_seconds": started - enqueued_at,
            "num_model_requests": 1, "images": image_details,
        }
        if effective == "noul":
            response["value"] = answer["value"]
        elif effective == "score":
            response["expected_score"] = sum(i * p for i, p in enumerate(probabilities))
        return response


engine = Engine()


@asynccontextmanager
async def lifespan(app):
    await asyncio.get_running_loop().run_in_executor(engine.executor, engine.initialize)
    yield
    engine.ready = False
    engine.executor.shutdown(wait=True)


app = FastAPI(title="Local JEV-27B-VL API", version="1.0.0", lifespan=lifespan,
              description="8-bit MLX System 1 decisions on this Mac. Text and base64 PNG/JPEG/WebP images. One GPU request executes at a time; up to four may be pending. This is a decision API, not a chat-completions API.")
app.add_middleware(BodyLimitMiddleware)
app.add_middleware(TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost"])


@app.get("/health")
async def health():
    return {"status": "ready" if engine.ready else "loading", "service": "local-jev-mlx", "model": MODEL_ID,
            "device": getattr(engine, "device", None), "model_memory_bytes": getattr(engine, "model_memory_bytes", None),
            "model_revision": getattr(engine, "revision", None), "pending_requests": engine.pending,
            "uptime_seconds": time.time() - engine.started, "max_input_tokens": MAX_INPUT_TOKENS}


@app.get("/v1/models")
async def models():
    return {"object": "list", "data": [{"id": MODEL_ID, "object": "model", "owned_by": "local", "capabilities": ["text_decisions", "image_decisions"]}]}


@app.post("/v1/decide")
async def decide(request: DecideRequest):
    if not engine.ready:
        raise HTTPException(503, "Model is loading")
    if engine.pending >= MAX_PENDING:
        raise HTTPException(429, "GPU queue is full; retry shortly", headers={"Retry-After": "2"})
    engine.pending += 1
    enqueued_at = time.perf_counter()
    future = asyncio.get_running_loop().run_in_executor(engine.executor, engine.decide, request, enqueued_at)

    def finished(_):
        engine.pending -= 1

    future.add_done_callback(finished)
    try:
        return await asyncio.shield(future)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    except Exception as exc:
        logging.getLogger("uvicorn.error").exception("JEV inference failed")
        raise HTTPException(500, "Inference failed; see local server log") from exc
