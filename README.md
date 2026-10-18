# JEV MLX API

Run **JEV-27B-VL 8-bit** locally on an Apple Silicon Mac, with a persistent HTTP API for text and image decisions. On supported M5 Macs, the recipe builds and selects the tested native INT8 runtime automatically. It reduced measured latency by **14–22% for long text and 7–15% for images** on an M5 Max; short Boolean requests were essentially unchanged.

```bash
git clone https://github.com/samuelcardillo/jev-mlx-api.git
cd jev-mlx-api
chmod +x recipe serve jev
./recipe --runtime auto
```

`auto` is the default, so `./recipe` works too. Setup installs pinned dependencies, downloads the **30.7 GB** model, chooses the runtime, and starts the server. Re-running it reuses completed installation work. Model weights, environments, build products and runtime state stay in the ignored `.jev/` directory.

The native path is experimental: it quantizes intermediate activations as well as using the existing 8-bit weights, and can change probabilities. Across 12 additional probes, every most-probable option matched released MLX; the largest probability change was **0.66 percentage points**, and the largest continuous-score change was **0.0077 on the 0–5 scale**. These are small synthetic checks, not a general accuracy guarantee. Use `./recipe --runtime stock` to select released MLX explicitly.

- **API:** `POST http://127.0.0.1:8765/v1/decide`
- **Interactive docs:** http://127.0.0.1:8765/docs
- **Readiness:** http://127.0.0.1:8765/health
- **Model listing:** http://127.0.0.1:8765/v1/models

## Requirements

- An Apple Silicon Mac with macOS and Metal support for the pinned MLX version.
- **64 GB unified memory recommended** for comfortable headroom. Tested on an M5 Max with 128 GB and macOS 26.6; smaller configurations are not validated by this recipe.
- At least **40 GB free disk space**, with additional room for download caches and the native source build.
- Apple's Command Line Tools (`git` and `/usr/bin/python3`). Install with `xcode-select --install` if missing.
- Internet access for the first installation. The complete installation runs offline afterward.

Native INT8 additionally requires an **M5-family Mac, macOS 26.2 or later, and an available macOS SDK 26.5 or later**. The first native installation compiles the pinned MLX source, so it takes longer than the stock installation. The recipe downloads no prebuilt experimental wheel from this repository.

| Setup mode | Behavior |
|---|---|
| `./recipe --runtime auto` | Select native INT8 when its hardware/toolchain requirements are met; otherwise select released MLX. |
| `./recipe --runtime stock` | Install/select the pinned released MLX runtime. |
| `./recipe --runtime int8` | Require the native runtime; report an error if its requirements are not met. |

The chosen mode is saved in `.jev/runtime.json`. Both `./serve` and `./jev` use that selection automatically. The stock environment remains in `.jev/.venv`; native INT8 has a separate `.jev/.venv-int8`, with its source checkout and build products under `.jev/native-build`. The native builder verifies its pinned inputs and runs numerical/attention checks before recording a successful installation.

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
./serve status   # Readiness, active runtime, model memory, and pending requests
./serve stop     # Graceful shutdown; releases GPU memory
./serve start    # Start the existing installation in the background
./serve logs     # Recent logs
./serve run      # Foreground mode; stop with Ctrl+C
```

The service binds to **127.0.0.1 only**, with no API key and no cross-origin browser access. Keep it local unless you add appropriate authentication and deployment controls. It does not install a login service; after restarting the Mac, run `./recipe`.

One model stays resident while the service is running: approximately **31 GB** for weights, plus inference allocations and allocator cache. One GPU worker serializes requests; health checks remain on the asynchronous HTTP event loop.

### Select or override the runtime

Stop the server before changing its runtime. To persistently select released MLX:

```bash
./serve stop
./recipe --runtime stock
./serve status
```

Use `./recipe --runtime auto` or `./recipe --runtime int8` to select native mode again on a supported Mac. `/health` reports the active `runtime`, `mlx_version`, and `mlx_qmm_int8` flag.

`JEV_RUNTIME=stock` or `JEV_RUNTIME=int8` overrides the saved selection for one launcher invocation, without installing anything or changing `runtime.json`. The requested runtime must already be installed:

```bash
./serve stop
JEV_RUNTIME=stock ./serve start

# After stopping the server, run a CLI request using an installed native runtime:
./serve stop
JEV_RUNTIME=int8 ./jev --state 'Please refund my damaged order.' \
  --question 'Is the customer asking for a refund?'
./serve start  # Uses the saved selection again.
```

For an independently managed environment, the advanced `JEV_SERVER_PYTHON` override remains available. It must contain the API dependencies and pinned MLX-VLM source. A manually supplied `MLX_QMM_INT8=1` requires a build implementing that feature; the flag alone cannot add native INT8 support to a released wheel. Normal use needs neither variable.

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

`./jev --benchmark` benchmarks the selected runtime. With a custom installation directory, replace `.jev` in the last command with `$JEV_HOME`. Generated results go to the installation's `results/` directory. The stock environment can run the HTTP smoke-test client against either server runtime. The API smoke suite checks Boolean/choice/score responses, both image representations, JSON state, validation failures, concurrent opposite decisions, and the OpenAPI schema. All **17 checks passed** against the tested native API.

The portable installer was also exercised in a fresh runtime directory: source build, kernel validation, repeated setup, CLI decision, all 17 API checks, and switching between stock and INT8 passed. Model files were reused for that test. [Installation validation record](benchmarks/m5-max-portable-install-validation.json).

Run the 36 CPU-only selection and lifecycle tests without loading the model:

```bash
/usr/bin/python3 -B -m unittest discover -s tests -q
```

## Native INT8 versus released MLX: paired measurement

The following direct comparison used an M5 Max with 128 GB unified memory, macOS 26.6, and the identical 8-bit model. Values are median decision latency; reduction is `1 - native_median / stock_median`.

| Input | Prompt tokens | Released MLX | Native INT8 | Less latency |
|---|---:|---:|---:|---:|
| Short Boolean, positive | 43 | 0.1482 s | 0.1471 s | 0.7% |
| Short Boolean, negative | 50 | 0.1535 s | 0.1528 s | 0.4% |
| Short choice | 76 | 0.1969 s | 0.1903 s | 3.4% |
| ~1K text context | 1,076 | 1.5452 s | 1.3319 s | 13.8% |
| ~4K text context | 4,148 | 6.2540 s | 5.3125 s | 15.1% |
| ~8K text context | 8,244 | 16.4437 s | 12.8471 s | 21.9% |
| 512×512 synthetic image | 308 | 0.6770 s | 0.6290 s | 7.1% |
| 1024×1024 synthetic image | 1,076 | 2.3359 s | 1.9755 s | 15.4% |

Each case used one excluded warmup per runtime and **five alternating A/B pairs**. Separate Python workers kept both models resident; GPU requests were serialized, and synchronized inference timing inside each worker excluded IPC, startup, cold compilation and HTTP transport. There was no cross-request prefix/KV cache. Both model copies being resident affects memory conditions. This was a working desktop, so background/thermal variation remains possible and small differences are inconclusive.

All eight synthetic scenarios retained their expected answers. Native peak active MLX allocation was **32.67 GB per worker**, with allocator cache additional; swap remained zero. Across the 12 extra Boolean/choice/score probes, all most-probable options matched, but four continuous scores changed. Raw float32 probabilities were compared before four-decimal rounding. These checks do not establish general accuracy, calibration, or robustness near a decision boundary. The custom JIT build can itself differ from released MLX even with the INT8 flag disabled.

- [Paired measurements and full probability readouts](benchmarks/m5-max-native-int8-paired.json)
- [Additional quality comparison, exact prompts and probability vectors](benchmarks/m5-max-native-int8-quality.json)
- [Small matrix-kernel checks and their coverage limits](benchmarks/m5-max-native-int8-kernels.json)
- [Provenance of the tested native build](benchmarks/m5-max-native-int8-build.json)

The [original released-MLX benchmark](benchmarks/m5-max-128gb.json) remains available as a **historical installation check**: three measured calls after warmup, with one model resident. Its absolute times must not be compared with the later paired measurements to infer a speedup; runtime and desktop conditions differed.

## Reproducible versions

| Component | Pin |
|---|---|
| Model | `nativ-community/JEV-27B-VL-MLX-8bit` |
| Model revision | `a871d5f8787b3d8ce9d618e7260e959393030c7b` |
| MLX-VLM source | `Lazarus-931/mlx-vlm@597941d0d4eb900f1d066eba8d89abf474c0dd42` |
| Python | `3.12.15` |
| uv | `0.13.1` |
| Released MLX | `0.32.4` |
| Native MLX source base | `ml-explore/mlx@36969c094354086e15d2a327a9c50feb20b9cf11` (`v0.32.4`) |
| MLX-VLM package | `0.7.7` |

All other package versions are pinned in [runtime-requirements.txt](runtime-requirements.txt). Setup verifies the upstream checkout and editable installation, and checks model-file sizes against [model-manifest.json](model-manifest.json). Hugging Face handles the download. Existing modified source is left untouched and reported.

The [native build manifest](native/manifest.json) pins the source, build requirements and three patch hashes: staged 8-bit QMM, matching-release metallib/JIT support, and a split-attention template-argument correction. The native builder reuses the matching **MLX 0.32.4** base metallib and compiles the changed kernels at runtime. Its build identity and successful validation are recorded under the installation directory. The checked-in benchmark provenance describes the actual tested build; a fresh build may have a different date-derived package version or wheel hash. This repository contains source patches and measurements, **not an experimental binary wheel**.

Weights, caches, credentials, environments, PID files, and logs are excluded from this repository. Fresh setup requires the pinned upstream artifacts to remain available.

## Credits and license

- [AutoTrust JEV-27B-VL](https://huggingface.co/autotrust/JEV-27B-VL)
- [nativ-community 8-bit MLX conversion](https://huggingface.co/nativ-community/JEV-27B-VL-MLX-8bit)
- [Lazarus-931's JEV support in MLX-VLM](https://github.com/Blaizzy/mlx-vlm/pull/2466)
- [MLX-VLM](https://github.com/Blaizzy/mlx-vlm) and [Apple MLX](https://github.com/ml-explore/mlx)

The recipe and API wrapper are [MIT licensed](LICENSE). The included MLX source patches retain [Apple's MIT license](native/MLX-LICENSE.txt). Downloaded model weights and third-party software retain their own licenses; the model card specifies Apache 2.0. Model weights and complete upstream checkouts are not redistributed here.
