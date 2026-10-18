"""Install the pinned Apple Silicon runtime and 8-bit JEV model."""
import argparse
import fcntl
import json
import os
import platform
import subprocess
import tarfile
import tempfile
import urllib.request

from paths import HERE, INSTALL, MODEL, PYTHON, SOURCE

UV = INSTALL / "bin/uv"
SOURCE_REPO = "https://github.com/Lazarus-931/mlx-vlm.git"
SOURCE_COMMIT = "597941d0d4eb900f1d066eba8d89abf474c0dd42"
SOURCE_VERSION = "0.7.7"
UV_VERSION = "0.13.1"
PYTHON_VERSION = "3.12.15"


def run(*args, **kwargs):
    return subprocess.run([str(x) for x in args], check=True, **kwargs)


def install():
    os.environ.update({
        "UV_CACHE_DIR": str(INSTALL / "cache/uv"),
        "UV_PYTHON_INSTALL_DIR": str(INSTALL / "python"),
        "HF_HOME": str(INSTALL / "cache/huggingface"),
    })
    if not UV.exists():
        UV.parent.mkdir(exist_ok=True)
        print("Downloading isolated uv runtime...", flush=True)
        with tempfile.TemporaryDirectory(prefix="uv-bootstrap-", dir=INSTALL) as temporary:
            archive = os.path.join(temporary, "uv.tar.gz")
            urllib.request.urlretrieve(
                f"https://github.com/astral-sh/uv/releases/download/{UV_VERSION}/uv-aarch64-apple-darwin.tar.gz",
                archive,
            )
            with tarfile.open(archive) as tar:
                member = tar.getmember("uv-aarch64-apple-darwin/uv")
                temporary_uv = UV.with_suffix(".tmp")
                temporary_uv.write_bytes(tar.extractfile(member).read())
            temporary_uv.chmod(0o755)
            temporary_uv.replace(UV)
    uv_version = subprocess.check_output([str(UV), "--version"], text=True).split()[1]
    if uv_version != UV_VERSION:
        raise SystemExit(f"Existing uv is {uv_version}; this recipe pins {UV_VERSION}. It was left untouched.")

    if not SOURCE.exists():
        # Rename only a complete checkout into place, so interrupted downloads
        # can be safely retried without treating a partial clone as installed.
        with tempfile.TemporaryDirectory(prefix="source-bootstrap-", dir=INSTALL) as temporary:
            checkout = os.path.join(temporary, "mlx-vlm")
            run("git", "clone", "--no-checkout", "--filter=blob:none", SOURCE_REPO, checkout)
            run("git", "checkout", "--detach", SOURCE_COMMIT, cwd=checkout)
            os.rename(checkout, SOURCE)
    source_head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=SOURCE, text=True).strip()
    source_changes = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"], cwd=SOURCE, text=True
    ).strip()
    if source_head != SOURCE_COMMIT or source_changes:
        raise SystemExit("Existing MLX-VLM source differs from the pinned recipe. It was left untouched; inspect it before continuing.")

    if not PYTHON.exists():
        run(UV, "venv", "--python", PYTHON_VERSION, INSTALL / ".venv")
    python_version = subprocess.check_output(
        [str(PYTHON), "-c", "import platform; print(platform.python_version())"], text=True
    ).strip()
    if python_version != PYTHON_VERSION:
        raise SystemExit(f"Existing Python is {python_version}; this recipe pins {PYTHON_VERSION}. It was left untouched.")
    expected = dict(
        line.split("==", 1)
        for line in (HERE / "runtime-requirements.txt").read_text().splitlines()
        if line and not line.startswith("#")
    )
    check_code = "import importlib.metadata as m,json; print(json.dumps({d.metadata['Name'].lower().replace('_','-'):d.version for d in m.distributions()}))"
    installed = json.loads(subprocess.check_output([str(PYTHON), "-c", check_code], text=True))
    if any(installed.get(name.lower().replace("_", "-")) != version for name, version in expected.items()):
        run(UV, "pip", "install", "--python", PYTHON, "--no-deps", "-r", HERE / "runtime-requirements.txt")

    # A same-version wheel or an editable install from a different checkout
    # must not bypass the source pin.
    editable_ok = False
    if installed.get("mlx-vlm") == SOURCE_VERSION:
        code = "import importlib.metadata as m; print(m.distribution('mlx-vlm').read_text('direct_url.json') or '{}')"
        direct_url = json.loads(subprocess.check_output([str(PYTHON), "-c", code], text=True))
        editable_ok = direct_url.get("url") == SOURCE.as_uri() and direct_url.get("dir_info", {}).get("editable") is True
    if not editable_ok:
        run(UV, "pip", "install", "--python", PYTHON, "--no-deps", "--reinstall", "-e", SOURCE)

    manifest = json.loads((HERE / "model-manifest.json").read_text())

    def missing_files():
        return [
            item["name"] for item in manifest["files"]
            if not (MODEL / item["name"]).is_file()
            or (MODEL / item["name"]).stat().st_size != item["size"]
        ]

    if missing_files():
        print("Downloading pinned 8-bit model (30.7 GB total; completed files are reused)...", flush=True)
        code = "from huggingface_hub import snapshot_download; import sys; snapshot_download(sys.argv[1], revision=sys.argv[2], local_dir=sys.argv[3], max_workers=8)"
        run(PYTHON, "-c", code, manifest["repo"], manifest["revision"], MODEL)
    if missing_files():
        raise SystemExit("Model files did not match the pinned manifest after download. Rerun setup to resume.")
    print("Pinned released runtime, source, and model are installed.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", choices=("auto", "stock", "int8"),
                        default=os.environ.get("JEV_RUNTIME", "auto"),
                        help="auto builds the tested INT8 runtime on supported M5 Macs; stock keeps released MLX")
    args = parser.parse_args()
    if args.runtime not in ("auto", "stock", "int8"):
        parser.error("JEV_RUNTIME must be auto, stock, or int8 during setup.")
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise SystemExit("This recipe requires an Apple Silicon Mac.")
    INSTALL.mkdir(parents=True, exist_ok=True)
    with (INSTALL / "setup.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        from native_build import install_native, support_reason
        from runtime import save_selection
        selected = args.runtime
        reason = support_reason() if selected != "stock" else None
        if selected == "int8" and reason:
            raise SystemExit("INT8 is unavailable: " + reason + " Use ./recipe --runtime stock for released MLX.")
        if selected == "auto":
            selected = "stock" if reason else "int8"
            if reason:
                print("Using released MLX: " + reason, flush=True)
        install()
        try:
            metadata = install_native(UV, PYTHON, SOURCE, INSTALL) if selected == "int8" else None
        except (RuntimeError, OSError, subprocess.SubprocessError) as exc:
            raise SystemExit("INT8 setup failed: " + str(exc) + "\nThe released runtime is preserved. Use ./recipe --runtime stock to select it.")
        save_selection(selected, metadata)
        print("Selected inference runtime: " + selected, flush=True)


if __name__ == "__main__":
    main()
