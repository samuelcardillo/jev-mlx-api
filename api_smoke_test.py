"""Exercise the running local API and save inspectable results."""
import base64
from concurrent.futures import ThreadPoolExecutor
import json
import os
import time
import urllib.error
import urllib.request

from paths import HERE, RESULTS
URL = os.environ.get("JEV_API_URL", "http://127.0.0.1:8765").rstrip("/")
results = []


def request(payload, expected_status=200):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(URL + "/v1/decide", data=data, headers={"Content-Type": "application/json"})
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=60) as response:
            status, body = response.status, json.load(response)
    except urllib.error.HTTPError as error:
        status, body = error.code, json.load(error)
    elapsed = time.perf_counter() - started
    assert status == expected_status, (status, body)
    if status == 200:
        assert abs(sum(body["probabilities"]) - 1) < 1e-8
        assert body["choice"] == body["options"][body["choice_index"]]
        assert body["usage"]["completion_tokens"] == 0
    return {"status": status, "http_seconds": elapsed, "response": body}


def record(name, result):
    results.append({"name": name, **result})
    print(name + ": passed", flush=True)


with urllib.request.urlopen(URL + "/health", timeout=3) as response:
    health = json.load(response)
assert health["status"] == "ready"
record("health", {"response": health})
yes = {"kind": "noul", "state": "My parcel is damaged. Please refund my payment.", "question": "Is the customer asking for a refund?"}
no = {**yes, "kind": "bool", "state": "The parcel arrived safely. I am happy and will keep it."}
for name, payload, expected in [("boolean_yes", yes, True), ("boolean_no", no, False)]:
    result = request(payload)
    assert result["response"]["value"] == expected
    record(name, result)
choice = {"kind": "choice", "state": "My credit card was charged twice for the same purchase.", "question": "Which department should handle this?", "options": ["billing", "shipping", "technical support"]}
result = request(choice)
assert result["response"]["options"] == choice["options"] and result["response"]["choice"] == "billing"
record("choice_order", result)
result = request({"kind": "score", "state": "I absolutely love this product. It is perfect!", "question": "Rate the sentiment from 0 (very negative) to 5 (very positive)."})
body = result["response"]
assert len(body["probabilities"]) == 6 and 0 <= body["expected_score"] <= 5
assert body["choice_index"] == max(range(6), key=body["probabilities"].__getitem__)
assert abs(body["expected_score"] - sum(i * p for i, p in enumerate(body["probabilities"]))) < 1e-8
record("score_contract", result)
image = "data:image/png;base64," + base64.b64encode((HERE / "fixtures/shapes-512.png").read_bytes()).decode()
vision = {"kind": "choice", "question": "Which shape is on the left side of the image?", "options": ["red square", "blue circle", "green triangle"]}
for name, state in [("image_part", [{"image": image}]), ("openai_image_part", [{"type": "image_url", "image_url": {"url": image}}])]:
    result = request({**vision, "state": state})
    assert result["response"]["choice"] == "red square"
    record(name, result)
result = request({**choice, "state": {"image": "invoice.png", "customer_message": "My credit card was charged twice. Please fix the duplicate charge."}})
assert result["response"]["choice"] == "billing"
record("structured_json_state", result)
for name, payload in [
    ("reject_single_choice", {**choice, "options": ["billing"]}),
    ("reject_257_choices", {**choice, "options": [str(i) for i in range(257)]}),
    ("reject_unknown_kind", {**yes, "kind": "chat"}),
    ("reject_thinking", {**yes, "thinking": "on"}),
    ("reject_remote_image", {**vision, "images": ["https://example.com/image.png"]}),
    ("reject_invalid_base64", {**vision, "images": ["data:image/png;base64,!!!"]}),
    ("reject_oversized_context", {**yes, "state": "word " * 20000}),
]:
    record(name, request(payload, 422))
with ThreadPoolExecutor(max_workers=2) as pool:
    calls = list(pool.map(request, [yes, no]))
assert calls[0]["response"]["value"] is True and calls[1]["response"]["value"] is False
record("concurrent_opposite_requests", {"responses": calls})
with urllib.request.urlopen(URL + "/openapi.json", timeout=3) as response:
    schema = json.load(response)
assert "/v1/decide" in schema["paths"]
record("openapi_schema", {"status": 200})
RESULTS.mkdir(parents=True, exist_ok=True)
(RESULTS / "api-test-results.json").write_text(json.dumps({"all_passed": True, "checks": results}, indent=2))
print(f"All {len(results)} API checks passed.", flush=True)
