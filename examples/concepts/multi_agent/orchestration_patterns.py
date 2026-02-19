"""Multi-agent orchestration pattern index and runner.

This file now acts as a lightweight launcher for focused pattern scripts in
`examples/concepts/multi_agent/patterns/`.

Usage:
    python examples/concepts/multi_agent/orchestration_patterns.py --list
    python examples/concepts/multi_agent/orchestration_patterns.py --pattern 02
    python examples/concepts/multi_agent/orchestration_patterns.py --pattern all
"""

from __future__ import annotations

import argparse
import runpy
from pathlib import Path

PATTERN_FILES = {
    "01": "01_sequential_pipeline.py",
    "02": "02_parallel_fanout_fanin.py",
    "03": "03_supervisor_worker.py",
    "04": "04_reflection_loop.py",
    "05": "05_debate_consensus.py",
}

PATTERN_DESCRIPTIONS = {
    "01": "Sequential pipeline",
    "02": "Parallel fan-out / fan-in",
    "03": "Supervisor / worker hierarchy",
    "04": "Reflection / self-critique",
    "05": "Debate / consensus",
}


def patterns_dir() -> Path:
    """Return the directory containing pattern scripts."""
    return Path(__file__).with_name("patterns")


def list_patterns() -> None:
    """Print available pattern IDs and script files."""
    print("Available orchestration patterns:")
    for key in sorted(PATTERN_FILES):
        filename = PATTERN_FILES[key]
        description = PATTERN_DESCRIPTIONS[key]
        print(f"  {key}: {description:<32} ({filename})")


def run_pattern(pattern_id: str) -> None:
    """Execute a single pattern script by ID."""
    pattern_file = patterns_dir() / PATTERN_FILES[pattern_id]
    if not pattern_file.exists():
        raise FileNotFoundError(f"Pattern script not found: {pattern_file}")

    print(f"\n=== Running pattern {pattern_id}: {PATTERN_DESCRIPTIONS[pattern_id]} ===")
    runpy.run_path(str(pattern_file), run_name="__main__")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SKTK multi-agent pattern examples.")
    parser.add_argument(
        "--pattern",
        default="all",
        choices=["all", *sorted(PATTERN_FILES.keys())],
        help="Pattern ID to run (default: all).",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available patterns and exit.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.list:
        list_patterns()
        return

    if args.pattern == "all":
        list_patterns()
        for pattern_id in sorted(PATTERN_FILES):
            run_pattern(pattern_id)
        print("\nAll orchestration patterns completed.")
        return

    run_pattern(args.pattern)


if __name__ == "__main__":
    main()
