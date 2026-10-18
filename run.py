#!/usr/bin/env python3
"""Local JEV decisions and a reproducible latency/memory smoke benchmark."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import resource
import statistics
import subprocess
import time
from datetime import datetime

from paths import HERE, INSTALL, MODEL, RESULTS, SOURCE
os.environ.setdefault("HF_HOME", str(INSTALL / "cache/huggingface"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import mlx.core as mx
from mlx_vlm import load, predict
from PIL import Image


def memory():
    return {
        "mlx_active_bytes": mx.get_active_memory(),
        "mlx_peak_bytes": mx.get_peak_memory(),
        "mlx_cache_bytes": mx.get_cache_memory(),
        "process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }


def call(model, processor, state, questions):
    mx.synchronize()
    start = time.perf_counter()
    result = predict(model, processor, state, questions)
    mx.synchronize()
    duration = time.perf_counter() - start
    return {"seconds": duration, "result": result, "memory": memory()}


def cases(processor):
    refund = {"answer": {"type": "bool", "instructions": "Is the customer asking for a refund?"}}
    out = [
        {"name": "short_refund_yes", "state": "The parcel arrived damaged and I want my money back.", "questions": refund, "expected": True},
        {"name": "short_refund_no", "state": "The parcel arrived safely. I love the product and want to keep it. Thank you!", "questions": refund, "expected": False},
        {"name": "short_topic_choice", "state": "My Wi-Fi router will not connect to the internet. How can I fix the connection?", "questions": {"answer": {"type": "choice", "instructions": "Which team should handle this request?", "criteria": {"billing": "payments, charges, and invoices", "technical_support": "troubleshooting hardware, software, and connectivity", "sales": "buying new products"}}}, "expected": "technical_support"},
    ]
    tokenizer = getattr(processor, "tokenizer", processor)
    filler = "Reference note: Warehouse staff checked the shelves, recorded inventory counts, and updated the delivery calendar. No customer request appears in this reference note.\n"
    for target in (1024, 4096, 8192):
        ids = tokenizer.encode(filler * (target // 10 + 1), add_special_tokens=False)[:target]
        state = tokenizer.decode(ids) + "\nEND REFERENCE NOTES.\nCustomer message: My order arrived damaged. Please refund my payment."
        out.append({"name": f"text_{target}_token_context", "state": state, "questions": refund, "expected": True, "synthetic_context_tokens": target})
    fixture_dir = HERE / "fixtures"
    for size in (512, 1024):
        path = fixture_dir / f"shapes-{size}.png"
        with Image.open(path) as source:
            image = source.convert("RGB")
        out.append({"name": f"vision_{size}x{size}", "state": ["Inspect the picture.", image], "questions": {"answer": {"type": "choice", "instructions": "Which shape is on the left side of the image?", "criteria": ["red square", "blue circle", "green triangle"]}}, "expected": "red square", "image_size": [size, size], "image_file": str(path.relative_to(HERE))})
    return out


def benchmark(model, processor, load_seconds, repeats):
    manifest = json.loads((HERE / "model-manifest.json").read_text())
    report = {
        "started_at": datetime.now().astimezone().isoformat(),
        "device": mx.device_info(),
        "macos": platform.mac_ver()[0],
        "python": platform.python_version(),
        "versions": {x: importlib.metadata.version(x) for x in ("mlx", "mlx-vlm", "transformers", "huggingface-hub")},
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=SOURCE, text=True).strip(),
        "model_repo": manifest["repo"],
        "model_revision": manifest["revision"],
        "model_path": "$JEV_HOME/model",
        "runtime": os.environ.get("JEV_ACTIVE_RUNTIME", "custom"),
        "mlx_qmm_int8": os.environ.get("MLX_QMM_INT8", "0"),
        "power": subprocess.check_output(["pmset", "-g", "batt"], text=True).splitlines()[0],
        "swap_before": subprocess.check_output(["sysctl", "vm.swapusage"], text=True).strip(),
        "load_seconds": load_seconds,
        "after_load_memory": memory(),
        "method": "One warmup and repeated measured calls per case; one question per call; explicit GPU synchronization; no cross-request KV/prefix cache. Timings include tokenization, image preprocessing, inference, and probability readout. Images are preloaded. Token throughput measures input processing, not generation. Synthetic correctness checks are not a general accuracy evaluation.",
        "measured_repeats": repeats,
        "cases": [],
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    report_path = RESULTS / "benchmark-results.json"
    report_path.write_text(json.dumps(report, indent=2))
    for case in cases(processor):
        print(f"Starting {case['name']} (warmup + {repeats} measured calls)", flush=True)
        warmup = call(model, processor, case["state"], case["questions"])
        mx.reset_peak_memory()
        runs = []
        for i in range(repeats):
            run = call(model, processor, case["state"], case["questions"])
            answer = run["result"]["answers"]["answer"]["value"]
            run["correct"] = answer == case["expected"]
            runs.append(run)
            print(json.dumps({"case": case["name"], "run": i + 1, "seconds": round(run["seconds"], 4), "input_tokens": run["result"]["usage"]["input_tokens"], "answer": answer, "correct": run["correct"], "mlx_peak_GB": round(run["memory"]["mlx_peak_bytes"] / 1e9, 3)}), flush=True)
        durations = [r["seconds"] for r in runs]
        row = {key: value for key, value in case.items() if key != "state"}
        if isinstance(case["state"], str):
            row["state"] = case["state"]
        row.update({"warmup": warmup, "runs": runs, "median_seconds": statistics.median(durations), "min_seconds": min(durations), "max_seconds": max(durations), "input_tokens": runs[0]["result"]["usage"]["input_tokens"], "all_correct": all(r["correct"] for r in runs), "peak_mlx_bytes": max(r["memory"]["mlx_peak_bytes"] for r in runs)})
        row["input_tokens_per_second"] = row["input_tokens"] / row["median_seconds"]
        report["cases"].append(row)
        report_path.write_text(json.dumps(report, indent=2))
    report["finished_at"] = datetime.now().astimezone().isoformat()
    report["swap_after"] = subprocess.check_output(["sysctl", "vm.swapusage"], text=True).strip()
    report["final_memory"] = memory()
    report_path.write_text(json.dumps(report, indent=2))
    print("Saved " + str(report_path), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--benchmark", action="store_true")
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--state", default="")
    p.add_argument("--state-file", type=Path)
    p.add_argument("--image", action="append", type=Path, default=[])
    p.add_argument("--question", help="Boolean question, or a choice question with --choices")
    p.add_argument("--choices", nargs="+")
    p.add_argument("--questions-file", type=Path, help="Named questions in JEV JSON format")
    args = p.parse_args()
    if not args.benchmark and not (args.question or args.questions_file):
        p.error("Supply --question, --questions-file, or --benchmark")
    if args.repeats < 1:
        p.error("--repeats must be positive")
    print("Loading JEV-27B-VL 8-bit on " + mx.device_info()["device_name"], flush=True)
    start = time.perf_counter()
    model, processor = load(str(MODEL), lazy=False)
    mx.synchronize()
    load_seconds = time.perf_counter() - start
    print(f"Model loaded in {load_seconds:.3f}s; MLX active memory {mx.get_active_memory() / 1e9:.3f} GB", flush=True)
    if args.benchmark:
        benchmark(model, processor, load_seconds, args.repeats)
        return
    state = args.state_file.read_text() if args.state_file else args.state
    if args.image:
        state = [state, *[Image.open(x).convert("RGB") for x in args.image]]
    questions = json.loads(args.questions_file.read_text()) if args.questions_file else {"answer": {"type": "choice" if args.choices else "bool", "instructions": args.question}}
    if args.choices and not args.questions_file:
        questions["answer"]["criteria"] = args.choices
    result = call(model, processor, state, questions)
    result["load_seconds"] = load_seconds
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
