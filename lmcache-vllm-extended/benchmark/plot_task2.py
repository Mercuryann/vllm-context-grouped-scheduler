#!/usr/bin/env python3
"""Plot Task 2 baseline vs context_grouped from results JSON."""

import subprocess
import sys
from pathlib import Path

if __name__ == "__main__":
    script = Path(__file__).resolve().parent / "run_task2.py"
    raise SystemExit(subprocess.call([sys.executable, str(script), "--plot-only", *sys.argv[1:]]))
