# JEV MLX API

Run **JEV-27B-VL 8-bit** locally on an Apple Silicon Mac, with a persistent HTTP API for decisions over text and images.

```bash
git clone https://github.com/samuelcardillo/jev-mlx-api.git
cd jev-mlx-api
chmod +x recipe serve jev
./recipe
```

The recipe installs an isolated, pinned Python/MLX environment, downloads the **30.7 GB** model, and starts the server. Re-running it reuses the installation. Model weights and runtime files stay in the ignored `.jev/` directory.

- **API:** `POST http://127.0.0.1:8765/v1/decide`
- **Interactive docs:** http://127.0.0.1:8765/docs
- **Readiness:** http://127.0.0.1:8765/health
- **Model listing:** http://127.0.0.1:8765/v1/models

## Requirements

- An Apple Silicon Mac with macOS and Metal support for the pinned MLX version.
- **64 GB unified memory recommended** for comfortable headroom. Tested on an M5 Max with 128 GB and macOS 26.6; smaller configurations are not validated by this recipe.
- At least **40 GB free disk space**, with additional room for download caches.
- Apple's Command Line Tools (`git` and `/usr/bin/python3`). Install with `xcode-select --install` if missing.
- Internet access for the first installation. The complete installation runs offline afterward.

This uses a community MLX conversion and the pinned `feat/jev` implementation. It exposes JEV's **System 1 typed decisions**, including probability readout. It does not implement free-form chat completions or System 2 thinking.

## Make a decision

```bash
curl -sS http://127.0.0.1:8765/v1/decide \
  -H 'Content-Type: application/json' \
  -d '{
    "kind": "choice",
    "state": "My credit card was charged twice.",
    "question": "Which department should handle this request?",
    "options": ["billing", "shipping", "technical support"]
  }'
```

Or use the included request file:

```bash
curl -sS http://127.0.0.1:8765/v1/decide \
  -H 'Content-Type: application/json' \
  --data-binary @api-example.json
```

| Kind | Input | Output |
|---|---|---|
| `noul` or `bool` | A Boolean question; omit `options` | `value`, plus probabilities for `false` and `true` |
| `choice` | 2–256 unique string `options` | Selected option and a probability for each option |
| `score` | A rubric in `question`; omit `options` | Six probabilities for 0–5, the most probable score, and `expected_score` |

`state` can be a string, a JSON object, or a list of text/image parts. Top-level JSON objects remain structured state.

### Response fields

- `options` and `probabilities` retain option order. Probabilities are normalized to sum to one.
- `choice_index`, `choice`, and `confidence` identify the selected option.
- Boolean requests also return `value`.
- Score requests also return `expected_score`. This is a weighted average; `choice` remains the most probable discrete label.
- `usage` reports prompt tokens and **zero completion tokens**: decisions do not generate text.
- `inference_seconds` includes image preparation, validation, tokenization, model execution, and readout.
- `queue_seconds` is time waiting for the GPU worker. `elapsed_seconds` includes queueing and inference, excluding HTTP transport.
- `images` reports original and processed dimensions.

The core request fields follow the upstream JEV `/v1/decide` shape. Unsupported fields such as `thinking` are rejected rather than silently ignored.

## Image example

Use base64 PNG, JPEG, or WebP data URLs. This example needs only Python's standard library:

```python
import base64
import json
from pathlib import Path
from urllib.request import Request, urlopen

image = Path("fixtures/shapes-512.png")
data_url = "data:image/png;base64," + base64.b64encode(image.read_bytes()).decode()
payload = {
    "kind": "choice",
    "state": ["Inspect this picture.", {"image": data_url}],
    "question": "Which shape is on the left side of the image?",
    "options": ["red square", "blue circle", "green triangle"],
}
request = Request(
    "http://127.0.0.1:8765/v1/decide",
    data=json.dumps(payload).encode(),
    headers={"Content-Type": "application/json"},
)
with urlopen(request, timeout=60) as response:
    print(json.dumps(json.load(response), indent=2))
```

Also accepted:

```json
{"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
```

Use this as a part inside the `state` list, or send `images: [data_url]` as a top-level request field. The HTTP API accepts image data, not remote URLs or local filesystem paths. The standalone CLI can read local image files.

## Manage the server

```bash
./serve status   # Readiness, PID, model memory, and pending requests
./serve stop     # Graceful shutdown; releases GPU memory
./serve start    # Start the existing installation in the background
./serve logs     # Recent logs
./serve run      # Foreground mode; stop with Ctrl+C
```

The service binds to **127.0.0.1 only**, with no API key and no cross-origin browser access. Keep it local unless you add appropriate authentication and deployment controls. It does not install a login service; after restarting the Mac, run `./recipe`.

One model stays resident while the service is running: approximately **31 GB** for weights, plus inference allocations and allocator cache. One GPU worker serializes requests; health checks remain on the asynchronous HTTP event loop.

### Runtime directory

By default all installation state lives in `.jev/`. To place it elsewhere:

```bash
export JEV_HOME="$PWD/../jev-runtime"
./recipe
```

Keep the same `JEV_HOME` setting when using `./serve` or `./jev`. A custom directory is outside the default `.gitignore` rule; keep it outside your repository or add it to your own ignore rules.

## Limits and errors

- At most **four pending requests**, including the active request. A full queue returns **429** with `Retry-After`.
- **16 MiB** request body; oversized requests return **413**.
- Up to **8 images**, **8 MiB decoded bytes per image**, and **20 megapixels per source image**.
- Images fit within **1024×1024** by default. Set `max_image_dimension` between 128 and 2048 to trade detail against latency. The response records processed dimensions.
- A conservative **16,384-token** request budget includes text, estimated image tokens, and framing allowance.
- Invalid inputs return **422**. Inference failures return **500** and are recorded in `.jev/api.log`.

## Standalone CLI and validation

```bash
./serve stop  # The standalone CLI loads its own model copy.

./jev --state 'Please refund my damaged order.' \
  --question 'Is the customer asking for a refund?'

./jev --image fixtures/shapes-512.png \
  --question 'Which shape is on the left?' \
  --choices 'red square' 'blue circle' 'green triangle'

./jev --benchmark --repeats 3
./serve start
.jev/.venv/bin/python api_smoke_test.py
```

With a custom runtime, replace `.jev` in the last command with `$JEV_HOME`. Generated results go to the runtime's `results/` directory. The API smoke suite checks Boolean/choice/score responses, both image representations, JSON state, validation failures, concurrent opposite decisions, and the OpenAPI schema.

## Measured on an M5 Max, 128 GB

Direct MLX measurements on macOS 26.6, plugged into power. Each scenario used one excluded warmup and three measured calls. One question per call; no cross-request prefix/KV cache.

| Input | Actual prompt tokens | Median decision latency |
|---|---:|---:|
| Short Boolean, positive | 43 | 0.143 s |
| Short Boolean, negative | 50 | 0.148 s |
| Short choice | 76 | 0.192 s |
| ~1K text context | 1,076 | 1.282 s |
| ~4K text context | 4,148 | 5.354 s |
| ~8K text context | 8,244 | 11.870 s |
| 512×512 synthetic image | 308 | 0.452 s |
| 1024×1024 synthetic image | 1,076 | 1.605 s |

Peak active MLX allocation was **32.38 GB**; the allocator retained another **7.73 GB** of cache at the end. No swap was used. Initial model materialization took **1.43 s**, with the OS file cache potentially warm. All 24 measured decisions matched the synthetic expected answers; this checks the installation, not general model accuracy. These timings exclude HTTP transport.

[Machine-readable benchmark](benchmarks/m5-max-128gb.json)

## Reproducible versions

| Component | Pin |
|---|---|
| Model | `nativ-community/JEV-27B-VL-MLX-8bit` |
| Model revision | `a871d5f8787b3d8ce9d618e7260e959393030c7b` |
| MLX-VLM source | `Lazarus-931/mlx-vlm@597941d0d4eb900f1d066eba8d89abf474c0dd42` |
| Python | `3.12.15` |
| uv | `0.13.1` |
| MLX | `0.32.4` |
| MLX-VLM package | `0.7.7` |

All other package versions are pinned in [runtime-requirements.txt](runtime-requirements.txt). Setup verifies the upstream checkout and editable installation, and checks model-file sizes against [model-manifest.json](model-manifest.json). Hugging Face handles the download. Existing modified source is left untouched and reported.

Weights, caches, credentials, environments, PID files, and logs are excluded from this repository. Fresh setup requires the pinned upstream artifacts to remain available.

## Credits and license

- [AutoTrust JEV-27B-VL](https://huggingface.co/autotrust/JEV-27B-VL)
- [nativ-community 8-bit MLX conversion](https://huggingface.co/nativ-community/JEV-27B-VL-MLX-8bit)
- [Lazarus-931's JEV support in MLX-VLM](https://github.com/Blaizzy/mlx-vlm/pull/2466)
- [MLX-VLM](https://github.com/Blaizzy/mlx-vlm) and [Apple MLX](https://github.com/ml-explore/mlx)

The recipe and API wrapper are [MIT licensed](LICENSE). Downloaded model weights and third-party software retain their own licenses; the model card specifies Apache 2.0. This repository does not redistribute those weights or upstream source.
