#!/usr/bin/env python3
"""Deprecated: plotting is built into run_task1.py (q1/q2/q3 auto-plot)."""

import sys

print(
    "plot_task1.py is no longer needed.\n"
    "Run:  python benchmark/run_task1.py q1|q2|q3\n"
    "Plots are written next to JSON in benchmark/results/.",
    file=sys.stderr,
)
sys.exit(1)
