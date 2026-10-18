"""Build and validate the optional pinned MLX INT8 backend in an isolated runtime.

The stock installation remains the source of the matching release's unchanged
base Metal library. Modified kernels compile through MLX's runtime JIT. No
runtime selection is made here; setup owns runtime.json after this succeeds.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import tempfile

from paths import HERE

NATIVE = HERE / "native"


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path):
    try:
        value = json.loads(Path(path).read_text())
        if not isinstance(value, dict):
            raise ValueError("expected a JSON object")
        return value
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"Cannot read native runtime metadata at {path}: {exc}") from exc


def _save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def _output(*args, cwd=None, env=None, timeout=30):
    return subprocess.check_output([str(x) for x in args], cwd=cwd, env=env,
                                   text=True, stderr=subprocess.PIPE, timeout=timeout).strip()


def _run(*args, **kwargs):
    return subprocess.run([str(x) for x in args], check=True, **kwargs)


def _version(value):
    numbers = re.findall(r"\d+", str(value))
    return tuple(int(x) for x in numbers[:3]) + (0,) * max(0, 3 - len(numbers))


def _cpu_brand():
    return _output("/usr/sbin/sysctl", "-n", "machdep.cpu.brand_string")


def _manifest():
    data = _json(NATIVE / "manifest.json")
    if data.get("schema_version") != 1:
        raise RuntimeError("Unsupported native manifest schema")
    for item in data["files"]:
        path = NATIVE / item["name"]
        if path.parent != NATIVE or not path.is_file() or _sha(path) != item["sha256"]:
            raise RuntimeError(f"Native recipe file does not match its manifest: {item['name']}")
    if _sha(HERE / "runtime-requirements.txt") != data["runtime_requirements_sha256"]:
        raise RuntimeError("Runtime requirements differ from the native manifest")
    return data


def build_id():
    """Return the verified manifest's SHA256, including pinned patch identities."""
    _manifest()
    return _sha(NATIVE / "manifest.json")


def _sdk():
    """Prefer the tested 26.5 SDK; accept another usable SDK >=26.5."""
    roots = {Path("/Library/Developer/CommandLineTools/SDKs")}
    candidates = set()
    try:
        developer = Path(_output("/usr/bin/xcode-select", "-p"))
        roots.add(developer / "SDKs")
        roots.add(developer / "Platforms/MacOSX.platform/Developer/SDKs")
        candidates.add(Path(_output("/usr/bin/xcrun", "--sdk", "macosx", "--show-sdk-path")))
        clang = _output("/usr/bin/xcrun", "--find", "clang++")
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError("A usable macOS SDK >=26.5 and Apple command line tools are required; install/select Xcode or Command Line Tools.") from exc
    for root in roots:
        if root.is_dir():
            candidates.update(root.glob("MacOSX*.sdk"))
    eligible = {}
    for candidate in candidates:
        try:
            path = candidate.resolve(strict=True)
            settings = json.loads((path / "SDKSettings.json").read_text())
            version = settings["Version"]
            if _version(version) >= (26, 5, 0):
                eligible[path] = version
        except (OSError, ValueError, KeyError):
            continue
    ordered = sorted(eligible.items(), key=lambda item: (0 if _version(item[1])[:2] == (26, 5) else 1, _version(item[1])))
    for path, version in ordered:
        # Link the same system frameworks used by MLX to detect SDK/linker
        # incompatibilities as well as C++ header problems. The temporary
        # executable is never run, and no GPU operation is performed.
        with tempfile.TemporaryDirectory(prefix="jev-sdk-probe-") as temporary:
            probe = subprocess.run(
                [clang, "-isysroot", str(path), "-mmacosx-version-min=26.2", "-std=c++20",
                 "-x", "c++", "-", "-framework", "Metal", "-framework", "Foundation",
                 "-o", str(Path(temporary) / "sdk-probe")],
                input="#include <cstddef>\n#include <Availability.h>\nint main() { return 0; }\n",
                text=True, capture_output=True, timeout=30,
            )
        if probe.returncode == 0:
            return {"path": str(path), "version": version, "clang": clang}
    raise RuntimeError("No usable macOS SDK >=26.5 was found. Install/select compatible Xcode or Command Line Tools; the tested SDK is 26.5.")


def support_reason():
    """None for supported M5/macOS/toolchain combinations; otherwise explain."""
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        return "Native INT8 requires an Apple Silicon Mac."
    if _version(platform.mac_ver()[0]) < (26, 2, 0):
        return "Native INT8 requires macOS 26.2 or newer."
    try:
        if not re.search(r"\bApple M5\b", _cpu_brand()):
            return "The tested native INT8 recipe is enabled for Apple M5 Macs; this Mac uses the stock runtime."
        _sdk()
    except (OSError, subprocess.SubprocessError, RuntimeError) as exc:
        return str(exc)
    return None


def _requirements():
    return dict(line.split("==", 1) for line in (HERE / "runtime-requirements.txt").read_text().splitlines()
                if line and not line.startswith("#") and not line.startswith(("mlx==", "mlx-metal==")))


def _inspect_python(python):
    # This deliberately imports metadata, not mlx.core: reuse checks do not
    # initialize Metal or run a model. Paths retain the virtualenv symlink.
    code = r'''
import hashlib, importlib.metadata as m, json, pathlib, platform
packages = {d.metadata['Name'].lower().replace('_', '-'): d.version for d in m.distributions()}
result = {'python_version': platform.python_version(), 'packages': packages}
try:
    vlm = m.distribution('mlx-vlm')
    result['mlx_vlm_direct_url'] = json.loads(vlm.read_text('direct_url.json') or '{}')
except m.PackageNotFoundError:
    result['mlx_vlm_direct_url'] = None
try:
    dist = m.distribution('mlx')
    binaries = {}
    for entry in dist.files or []:
        name = str(entry)
        if name.startswith('mlx/') and (name.endswith(('.so', '.dylib')) or name == 'mlx/lib/mlx.metallib'):
            path = pathlib.Path(dist.locate_file(entry))
            digest = hashlib.sha256()
            with path.open('rb') as f:
                for block in iter(lambda: f.read(1024 * 1024), b''):
                    digest.update(block)
            binaries[name] = digest.hexdigest()
    result['binaries'] = binaries
    result['metallib_path'] = str(dist.locate_file('mlx/lib/mlx.metallib'))
except m.PackageNotFoundError:
    result['binaries'] = {}
print(json.dumps(result))
'''
    return json.loads(_output(python, "-c", code, timeout=60))


def _check_source(source, manifest):
    source = Path(source).resolve()
    try:
        head = _output("git", "rev-parse", "HEAD", cwd=source)
        changes = _output("git", "status", "--porcelain", "--untracked-files=no", cwd=source)
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError("The pinned MLX-VLM checkout is unavailable") from exc
    if head != manifest["mlx_vlm_commit"] or changes:
        raise RuntimeError("MLX-VLM source differs from the pinned recipe; it was left untouched.")


def _check_dependencies(info, source, manifest, native=False):
    if info["python_version"] != manifest["python_version"]:
        raise RuntimeError("Native runtime Python version differs from the recipe pin")
    required = _requirements()
    required["mlx-vlm"] = manifest["mlx_vlm_version"]
    missing = [name for name, version in required.items() if info["packages"].get(name) != version]
    direct = info.get("mlx_vlm_direct_url") or {}
    if missing or direct.get("url") != Path(source).resolve().as_uri() or direct.get("dir_info", {}).get("editable") is not True:
        raise RuntimeError("Runtime dependencies or editable MLX-VLM source do not match the recipe: " + ", ".join(missing or ["mlx-vlm source"]))
    if native and "mlx-metal" in info["packages"]:
        raise RuntimeError("The monolithic native environment must not contain mlx-metal")


def _checkout_state(checkout, expected_id, manifest):
    marker = checkout / ".git/jev-native-source.json"
    if not marker.is_file():
        raise RuntimeError(f"Partial or unmanaged native checkout at {checkout}; inspect/move it before retrying. It was left untouched.")
    data = _json(marker)
    dirty = _output("git", "diff", "--name-only", cwd=checkout)
    extra = _output("git", "ls-files", "--others", "--exclude-standard", cwd=checkout)
    tree = _output("git", "write-tree", cwd=checkout)
    head = _output("git", "rev-parse", "HEAD", cwd=checkout)
    if (data.get("build_id") != expected_id or head != manifest["mlx_base_commit"]
            or dirty or extra or tree != data.get("patched_tree")):
        raise RuntimeError(f"Native checkout has partial or user changes at {checkout}. It was left untouched; inspect/move it before retrying.")
    return data


def _prepare_checkout(build_root, expected_id, manifest):
    checkout = build_root / ("mlx-" + expected_id[:16])
    if checkout.exists():
        _checkout_state(checkout, expected_id, manifest)
        return checkout
    with tempfile.TemporaryDirectory(prefix="source-bootstrap-", dir=build_root) as temp:
        stage = Path(temp) / "mlx"
        _run("git", "clone", "--no-checkout", "--filter=blob:none", manifest["mlx_repository"], stage)
        _run("git", "checkout", "--detach", manifest["mlx_base_commit"], cwd=stage)
        for name in manifest["patch_order"]:
            _run("git", "apply", "--check", NATIVE / name, cwd=stage)
            _run("git", "apply", NATIVE / name, cwd=stage)
        _run("git", "add", "--all", cwd=stage)
        state = {"build_id": expected_id, "base_commit": manifest["mlx_base_commit"],
                 "patched_tree": _output("git", "write-tree", cwd=stage)}
        _save(stage / ".git/jev-native-source.json", state)
        stage.rename(checkout)
    _checkout_state(checkout, expected_id, manifest)
    return checkout


def _native_python(install):
    return Path(install) / ".venv-int8/bin/python"


def validate_installed(INSTALL, SOURCE, PYTHON=None):
    """Verify a completed installation without GPU work; None if not installed.

    Modified completed installations raise instead of being silently repaired.
    PYTHON is accepted for the setup interface; the native interpreter is fixed.
    """
    install, source = Path(INSTALL).resolve(), Path(SOURCE).resolve()
    marker = install / "native-runtime.json"
    if not marker.exists():
        return None
    manifest, expected_id = _manifest(), build_id()
    data = _json(marker)
    python = _native_python(install)
    if data.get("schema_version") != 1 or data.get("build_id") != expected_id or data.get("python_path") != str(python):
        raise RuntimeError("Native runtime marker differs from this recipe. Inspect/move .venv-int8 and native-runtime.json before rebuilding; they were left untouched.")
    if not python.is_file() or not os.access(python, os.X_OK):
        raise RuntimeError("Native interpreter is missing or not executable")
    _check_source(source, manifest)
    checkout = install / "native-build" / ("mlx-" + expected_id[:16])
    _checkout_state(checkout, expected_id, manifest)
    wheel = install / "native-build/wheels" / expected_id / data.get("wheel_name", "")
    if wheel.parent != install / "native-build/wheels" / expected_id or not wheel.is_file() or _sha(wheel) != data.get("wheel_sha256"):
        raise RuntimeError("The retained native wheel is missing or differs from its installation marker")
    info = _inspect_python(python)
    _check_dependencies(info, source, manifest, native=True)
    if (info["packages"].get("mlx") != data.get("mlx_version")
            or info.get("binaries") != data.get("installed_binaries_sha256")
            or info.get("python_version") != data.get("python_version")):
        raise RuntimeError("Installed native MLX files or version differ from the validated runtime; they were left untouched.")
    result_path = install / "native-build" / ("validation-" + expected_id + ".json")
    if not result_path.is_file() or _sha(result_path) != data.get("validation_sha256"):
        raise RuntimeError("Native kernel validation receipt is missing or changed")
    result = _json(result_path)
    if not result.get("passed") or not result.get("eligible_output_changed") or not result.get("attention_check", {}).get("passed"):
        raise RuntimeError("Native kernel/attention validation did not pass")
    return data


def _builder(uv, stock_python, build_root, manifest, env):
    builder = build_root / ".venv"
    python = builder / "bin/python"
    marker = builder / "jev-builder.json"
    if builder.exists() and not marker.is_file():
        raise RuntimeError(f"Partial or unmanaged builder environment at {builder}; inspect/move it before retrying.")
    if not builder.exists():
        _run(uv, "venv", "--python", stock_python, builder, env=env)
        _save(marker, {"python_version": manifest["python_version"], "managed_by": "jev-native-recipe"})
    info = _inspect_python(python)
    if info["python_version"] != manifest["python_version"]:
        raise RuntimeError("Existing native builder has a different Python version; it was left untouched.")
    expected = dict(line.split("==", 1) for line in (NATIVE / "build-requirements.txt").read_text().splitlines() if line)
    if any(info["packages"].get(name) != version for name, version in expected.items()):
        _run(uv, "pip", "install", "--python", python, "--no-deps", "-r", NATIVE / "build-requirements.txt", env=env)
    if not _output(builder / "bin/cmake", "--version", env=env).startswith("cmake version " + expected["cmake"]):
        raise RuntimeError("The pinned CMake executable is unavailable")
    return python


def _cmake_string(value):
    # CMake bracket quoting handles spaces, quotes, backslashes and semicolons.
    delimiter = "="
    while "]" + delimiter + "]" in str(value):
        delimiter += "="
    return "[" + delimiter + "[" + str(value) + "]" + delimiter + "]"


def _build_wheel(builder_python, checkout, build_root, expected_id, metallib, sdk, manifest, env):
    wheel_dir = build_root / "wheels" / expected_id
    receipt = wheel_dir / "build.json"
    config = {"build_id": expected_id, "sdk": sdk, "metallib_sha256": _sha(metallib)}
    if receipt.exists():
        existing = _json(receipt)
        path = wheel_dir / existing.get("wheel_name", "")
        if (existing.get("build_id") == expected_id and path.parent == wheel_dir and path.is_file()
                and _sha(path) == existing.get("wheel_sha256")):
            return path, existing
        raise RuntimeError("Partial or changed native wheel receipt; inspect native-build/wheels before retrying")
    wheel_dir.mkdir(parents=True, exist_ok=True)
    if list(wheel_dir.glob("*.whl")):
        raise RuntimeError("A native wheel exists without a completed build receipt; inspect/move it before retrying")
    # setup.py splits CMAKE_ARGS on spaces. Use a relative initial-cache file
    # instead of putting user/SDK paths into that string.
    cmake_dir = checkout / "build/native/mlx.core"
    cmake_dir.mkdir(parents=True, exist_ok=True)
    options = {"MLX_METAL_JIT": "ON", "MLX_PREBUILT_METALLIB": str(metallib),
               "MLX_PREBUILT_METAL_VERSION": "400", "CMAKE_OSX_DEPLOYMENT_TARGET": "26.2",
               "CMAKE_OSX_SYSROOT": sdk["path"], "CMAKE_OSX_ARCHITECTURES": "arm64",
               "MLX_BUILD_FRONTEND_PACKAGE": "OFF", "MLX_USE_CCACHE": "OFF"}
    initial_cache = "\n".join(f"set({name} {_cmake_string(value)} CACHE STRING \"JEV native recipe\" FORCE)" for name, value in options.items())
    (cmake_dir / "native-options.cmake").write_text(initial_cache + "\n")
    build_env = dict(env)
    for name in ("MLX_BUILD_FRONTEND_PACKAGE", "MLX_BUILD_BACKEND_PACKAGE", "PYPI_RELEASE", "DEV_RELEASE", "DEBUG", "ARCHFLAGS", "CMAKE_GENERATOR"):
        build_env.pop(name, None)
    build_env.update({"SDKROOT": sdk["path"], "MACOSX_DEPLOYMENT_TARGET": "26.2",
                      "CMAKE_ARGS": "-C native-options.cmake",
                      "CMAKE_BUILD_PARALLEL_LEVEL": str(min(8, os.cpu_count() or 1)),
                      "PATH": str(builder_python.parent) + os.pathsep + env.get("PATH", "")})
    print("Building the pinned native MLX wheel; this can take several minutes...", flush=True)
    # Set the parent build command's temp directory. bdist_wheel reinitializes
    # build_ext, which would discard a temp override supplied only to build_ext.
    _run(builder_python, "setup.py", "build", "--build-temp", "build/native", "bdist_wheel", "--dist-dir", wheel_dir,
         cwd=checkout, env=build_env)
    wheels = list(wheel_dir.glob("mlx-*.whl"))
    if len(wheels) != 1:
        raise RuntimeError("Native build did not produce exactly one monolithic MLX wheel")
    config.update({"wheel_name": wheels[0].name, "wheel_sha256": _sha(wheels[0]), "built_at": datetime.now(timezone.utc).isoformat()})
    _save(receipt, config)
    return wheels[0], config


def install_native(UV, PYTHON, SOURCE, INSTALL):
    """Build/validate the isolated backend and return native-runtime metadata."""
    reason = support_reason()
    if reason:
        raise RuntimeError(reason)
    manifest, expected_id = _manifest(), build_id()
    install, source = Path(INSTALL).resolve(), Path(SOURCE).resolve()
    # Keep interpreter paths lexical so a venv symlink is not bypassed.
    uv, stock_python = Path(UV).absolute(), Path(PYTHON).absolute()
    existing = validate_installed(install, source, stock_python)
    if existing is not None:
        return existing
    _check_source(source, manifest)
    stock = _inspect_python(stock_python)
    _check_dependencies(stock, source, manifest)
    if stock["packages"].get("mlx") != manifest["mlx_base_version"] or stock["packages"].get("mlx-metal") != manifest["mlx_base_version"]:
        raise RuntimeError("Build the pinned stock MLX 0.32.4 environment before the native backend")
    metallib = Path(stock["metallib_path"])
    if not metallib.is_file() or _sha(metallib) != manifest["stock_metallib_sha256"]:
        raise RuntimeError("Stock MLX Metal library differs from the tested 0.32.4 artifact; it was left untouched")
    build_root = install / "native-build"
    build_root.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env.update({"UV_CACHE_DIR": str(install / "cache/uv"), "UV_PYTHON_INSTALL_DIR": str(install / "python"),
                "PYTHONNOUSERSITE": "1", "TOKENIZERS_PARALLELISM": "false"})
    env.pop("PYTHONPATH", None)
    builder_python = _builder(uv, stock_python, build_root, manifest, env)
    checkout = _prepare_checkout(build_root, expected_id, manifest)
    sdk = _sdk()
    wheel, wheel_receipt = _build_wheel(builder_python, checkout, build_root, expected_id, metallib, sdk, manifest, env)
    native_python = _native_python(install)
    native_env = native_python.parent.parent
    progress = build_root / "runtime-in-progress.json"
    if native_env.exists():
        if not progress.is_file() or _json(progress).get("build_id") != expected_id:
            raise RuntimeError(f"Partial or unmanaged native environment at {native_env}; inspect/move it before retrying. It was left untouched.")
    else:
        _save(progress, {"build_id": expected_id, "python_path": str(native_python)})
        _run(uv, "venv", "--python", stock_python, native_env, env=env)
    native_requirements = build_root / "runtime-requirements-native.txt"
    native_requirements.write_text("\n".join(name + "==" + version for name, version in _requirements().items()) + "\n")
    _run(uv, "pip", "install", "--python", native_python, "--no-deps", "-r", native_requirements, env=env)
    _run(uv, "pip", "install", "--python", native_python, "--no-deps", "--reinstall", wheel, env=env)
    _run(uv, "pip", "install", "--python", native_python, "--no-deps", "--reinstall", "--build-constraints", NATIVE / "build-requirements.txt", "-e", source, env=env)
    info = _inspect_python(native_python)
    _check_dependencies(info, source, manifest, native=True)
    core = {name: value for name, value in info["binaries"].items() if name.startswith("mlx/core") and name.endswith(".so")}
    if len(core) != 1 or "mlx/lib/libmlx.dylib" not in info["binaries"]:
        raise RuntimeError("The native wheel did not install its core and Metal backend")
    validation = build_root / ("validation-" + expected_id + ".json")
    print("Validating native INT8 activity, numerical error and wide-head attention...", flush=True)
    _run(native_python, NATIVE / "kernel_check.py", "--out", validation, env=env)
    result = _json(validation)
    if not result.get("passed") or not result.get("eligible_output_changed") or not result.get("attention_check", {}).get("passed"):
        raise RuntimeError("Native validation failed; runtime selection was not changed")
    _checkout_state(checkout, expected_id, manifest)
    data = {"schema_version": 1, "runtime": "int8", "build_id": expected_id,
            "python_path": str(native_python), "python_version": info["python_version"],
            "mlx_version": info["packages"]["mlx"], "mlx_base_commit": manifest["mlx_base_commit"],
            "wheel_name": wheel.name, "wheel_sha256": _sha(wheel),
            "installed_mlx_core_sha256": next(iter(core.values())),
            "installed_binaries_sha256": info["binaries"],
            "source_commit": manifest["mlx_vlm_commit"], "source_path": str(source),
            "validation": result, "validation_sha256": _sha(validation),
            "sdk": sdk, "build_receipt": wheel_receipt,
            "installed_at": datetime.now(timezone.utc).isoformat()}
    _save(install / "native-runtime.json", data)
    validated = validate_installed(install, source, stock_python)
    progress.unlink(missing_ok=True)
    return validated
