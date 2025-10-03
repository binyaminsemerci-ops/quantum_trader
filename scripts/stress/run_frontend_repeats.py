"""Legacy wrapper that delegates frontend repeat runs to harness.run_frontend_repeats().

This keeps existing invocation points stable while centralizing logic in harness.py.
"""
import argparse
from pathlib import Path
import importlib.util
import sys


def _load_harness_module():
    # Load harness.py from the same directory (robust when running the script directly)
    p = Path(__file__).resolve().parent / "harness.py"
    spec = importlib.util.spec_from_file_location("stress.harness", str(p))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--count", type=int, default=100)
    p.add_argument("--start-at", type=int, default=1)
    args = p.parse_args()
    harness = _load_harness_module()
    harness.run_frontend_repeats(count=args.count, start_at=args.start_at)


if __name__ == '__main__':
    main()
