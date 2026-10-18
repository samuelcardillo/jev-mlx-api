"""Shared recipe paths; runtime data stays outside tracked source files."""
import argparse
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
INSTALL = Path(os.environ.get("JEV_HOME", str(HERE / ".jev"))).expanduser().resolve()
SOURCE = INSTALL / "mlx-vlm"
PYTHON = INSTALL / ".venv/bin/python"
MODEL = INSTALL / "model"
RESULTS = INSTALL / "results"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", action="store_true", help="Print the isolated Python path")
    args = parser.parse_args()
    print(PYTHON if args.python else INSTALL)
