#!/usr/bin/env python3
"""Run a deterministic learning-curve grid through the standard trainer."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(
        description="Repeat run_train.py over training fractions and subset seeds; pass trainer arguments after --."
    )
    parser.add_argument("--fractions", type=float, nargs="+", default=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0])
    parser.add_argument("--fraction-seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--dry-run", action="store_true")
    args, trainer_args = parser.parse_known_args()
    if trainer_args and trainer_args[0] == "--":
        trainer_args = trainer_args[1:]
    if not trainer_args:
        parser.error("pass the normal run_train.py arguments after --")
    for fraction in args.fractions:
        if not 0 < fraction <= 1:
            parser.error(f"invalid fraction {fraction}; expected 0 < fraction <= 1")
    for fraction_seed in args.fraction_seeds:
        for fraction in args.fractions:
            command = [
                args.python,
                str(ROOT / "run_train.py"),
                *trainer_args,
                "--train_fraction",
                str(fraction),
                "--fraction_seed",
                str(fraction_seed),
            ]
            print(" ".join(command), flush=True)
            if not args.dry_run:
                subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
