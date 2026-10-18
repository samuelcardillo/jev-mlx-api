"""Select the installed inference runtime for the API and command-line client."""
import hashlib
import json
import os
from pathlib import Path
import sys

from paths import HERE, INSTALL, PYTHON, SOURCE

SELECTION = INSTALL / "runtime.json"


def executable(path):
    # Keep a venv's Python symlink: resolving it would bypass the environment.
    path = Path(path).expanduser()
    if not path.is_absolute() or not path.is_file() or not os.access(path, os.X_OK):
        raise SystemExit("Inference Python must be an absolute path to an executable interpreter: " + str(path))
    return path


def select_runtime(server=False):
    if server and "JEV_SERVER_PYTHON" in os.environ:
        return {"name": "custom", "python": executable(os.environ["JEV_SERVER_PYTHON"]),
                "int8_flag": os.environ.get("MLX_QMM_INT8", "0")}
    try:
        saved = json.loads(SELECTION.read_text()) if SELECTION.exists() else {}
    except (OSError, ValueError) as exc:
        raise SystemExit("Cannot read runtime selection; rerun ./recipe: " + str(exc))
    if not isinstance(saved, dict):
        raise SystemExit("Invalid runtime selection; rerun ./recipe.")
    name = os.environ.get("JEV_RUNTIME", saved.get("runtime", "stock"))
    if name not in ("stock", "int8"):
        raise SystemExit("JEV_RUNTIME must be stock or int8.")
    if name == "stock":
        return {"name": name, "python": executable(PYTHON), "int8_flag": "0"}
    from native_build import build_id, validate_installed
    marker_path = INSTALL / "native-runtime.json"
    try:
        marker = json.loads(marker_path.read_text())
    except (OSError, ValueError):
        raise SystemExit("The INT8 runtime is not installed. Run ./recipe --runtime int8, or use JEV_RUNTIME=stock.")
    if not isinstance(marker, dict):
        raise SystemExit("Invalid INT8 runtime metadata. Rerun ./recipe --runtime int8.")
    if marker.get("build_id") != build_id():
        raise SystemExit("The installed INT8 runtime is from a different recipe revision. Rerun ./recipe --runtime int8.")
    python = INSTALL / ".venv-int8/bin/python"
    if marker.get("python_path") != str(python):
        raise SystemExit("The INT8 runtime location has changed. Rerun ./recipe --runtime int8.")
    python = executable(python)
    try:
        if validate_installed(INSTALL, SOURCE, PYTHON) is None:
            raise RuntimeError("The native installation marker is missing.")
    except RuntimeError as exc:
        raise SystemExit(
            "INT8 runtime verification failed: " + str(exc)
            + "\nUse JEV_RUNTIME=stock for released MLX, or inspect the INT8 installation before rerunning ./recipe --runtime int8."
        ) from exc
    return {"name": name, "python": python, "int8_flag": "1", "build_id": marker["build_id"]}


def child_environment(selected):
    env = dict(os.environ)
    env.update({"JEV_HOME": str(INSTALL), "JEV_ACTIVE_RUNTIME": selected["name"],
                "MLX_QMM_INT8": selected["int8_flag"], "HF_HUB_OFFLINE": "1",
                "TOKENIZERS_PARALLELISM": "false", "PYTHONUNBUFFERED": "1"})
    return env


def signature(selected):
    return hashlib.sha256(json.dumps({key: str(value) for key, value in selected.items()}, sort_keys=True).encode()).hexdigest()


def save_selection(name, metadata=None):
    value = {"schema_version": 1, "runtime": name}
    if name == "int8":
        value["build_id"] = metadata["build_id"]
    temporary = SELECTION.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(SELECTION)


if __name__ == "__main__":
    selected = select_runtime()
    command = [str(selected["python"]), str(HERE / "run.py"), *sys.argv[1:]]
    os.execve(command[0], command, child_environment(selected))
