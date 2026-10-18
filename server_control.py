"""Start, stop, or inspect this recipe's local JEV server."""
import argparse
import fcntl
import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

from paths import HERE, INSTALL, PYTHON
PID_FILE = INSTALL / "api-process.json"
LOG_FILE = INSTALL / "api.log"
BASE_URL = "http://127.0.0.1:8765"


def health():
    try:
        with urllib.request.urlopen(BASE_URL + "/health", timeout=2) as r:
            return json.load(r)
    except (OSError, ValueError):
        return None


def owned_process():
    if not PID_FILE.exists():
        return None
    info = json.loads(PID_FILE.read_text())
    result = subprocess.run(["ps", "-p", str(info["pid"]), "-o", "args="], capture_output=True, text=True)
    if result.returncode == 0 and "uvicorn api:app" in result.stdout and str(HERE) in result.stdout:
        return info["pid"]
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", nargs="?", choices=["start", "stop", "status", "logs", "run"], default="start")
    args = parser.parse_args()
    lifecycle_lock = None
    if args.action in ("start", "stop"):
        INSTALL.mkdir(parents=True, exist_ok=True)
        lifecycle_lock = (INSTALL / "api-control.lock").open("a")
        fcntl.flock(lifecycle_lock, fcntl.LOCK_EX)
    command = [str(PYTHON), "-m", "uvicorn", "api:app", "--app-dir", str(HERE), "--host", "127.0.0.1", "--port", "8765", "--workers", "1", "--log-level", "info"]
    if args.action == "run":
        os.execv(command[0], command)
    if args.action == "logs":
        print(LOG_FILE.read_text()[-16000:] if LOG_FILE.exists() else "No log yet.")
        return
    pid = owned_process()
    if args.action == "stop":
        if not pid:
            print("No managed JEV server is running.")
            return
        os.kill(pid, signal.SIGTERM)
        for _ in range(120):
            if not owned_process():
                PID_FILE.unlink(missing_ok=True)
                print("JEV server stopped; GPU memory released.")
                return
            time.sleep(0.25)
        raise SystemExit("Server is still finishing a request; check status again shortly.")
    status = health()
    if args.action == "status":
        print(json.dumps({"managed_pid": pid, "url": BASE_URL, "health": status}, indent=2))
        return
    if status and status.get("service") == "local-jev-mlx":
        print("JEV is already ready at " + BASE_URL)
        return
    if not pid:
        with socket.socket() as sock:
            try:
                sock.bind(("127.0.0.1", 8765))
            except OSError:
                raise SystemExit("Port 8765 is occupied by another service; it was left untouched.")
        env = dict(os.environ)
        env.update({"JEV_HOME": str(INSTALL), "HF_HUB_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false", "PYTHONUNBUFFERED": "1"})
        with LOG_FILE.open("a") as log:
            process = subprocess.Popen(command, cwd=HERE, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        pid = process.pid
        temporary_pid_file = PID_FILE.with_suffix(".tmp")
        temporary_pid_file.write_text(json.dumps({"pid": pid, "started_at": time.time(), "url": BASE_URL}, indent=2))
        temporary_pid_file.replace(PID_FILE)
        print(f"Starting JEV (PID {pid}); loading model...", flush=True)
    for _ in range(120):
        status = health()
        if status and status.get("service") == "local-jev-mlx" and status.get("status") == "ready":
            print("Ready: " + BASE_URL + "/v1/decide\nInteractive docs: " + BASE_URL + "/docs", flush=True)
            return
        if not owned_process():
            print(LOG_FILE.read_text()[-8000:] if LOG_FILE.exists() else "Server exited.", file=sys.stderr)
            raise SystemExit(1)
        time.sleep(1)
    raise SystemExit("Startup is still in progress. Inspect ./serve logs and ./serve status.")


if __name__ == "__main__":
    main()
