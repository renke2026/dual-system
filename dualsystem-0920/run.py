# -*- coding: utf-8 -*-
"""Simulation entry point: runs the simulation (scripts/main.py).

Data preparation (pricing / agents / NPCs) is handled by generate.py.
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import CONFIG  # noqa: E402


def _env(value):
    """Coerce a value to a string for an env var; None means no override."""
    return None if value is None else str(value)


def run_stage(name, script, extra_env=None):
    """Run a stage script under scripts/ using the current interpreter."""
    script_path = PROJECT_ROOT / "scripts" / script
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    print(f"\n{'='*70}\n▶ Stage: {name}  (python scripts/{script})\n{'='*70}")
    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=str(PROJECT_ROOT),
        env=env,
    )
    if result.returncode != 0:
        print(f'❌ Stage "{name}" failed (exit code {result.returncode}).')
        sys.exit(result.returncode)
    print(f'✅ Stage "{name}" completed.')


def build_parser():
    p = argparse.ArgumentParser(description="Simulation entry point (generate data with generate.py first)")
    p.add_argument("--case", type=str, default=None, help="Road network case (nanjing / 12nodes), overrides config.paths.case_name")
    p.add_argument("--days", type=int, default=None, help="Simulation days (overrides config.simulation.days)")
    p.add_argument("--gui", dest="gui", action="store_true", default=None, help="Enable SUMO GUI")
    p.add_argument("--no-gui", dest="gui", action="store_false", help="Disable SUMO GUI")
    p.add_argument("--concurrency", type=int, default=None, help="Max LLM concurrency")
    p.add_argument("--seed", type=int, default=None, help="Random seed (default 42)")
    p.add_argument("--memory", dest="memory", action="store_true", default=None, help="Enable memory system (overrides default no-memory)")
    p.add_argument("--no-memory", dest="memory", action="store_false", help="Disable memory system (default)")
    p.add_argument("--baseline", type=str, choices=["full", "event_llm", "mnl"], default=None,
                   help="Decision engine: full (dual-system, default) / event_llm (Baseline A event-triggered LLM) / mnl (Baseline B conventional MNL)")

    p.add_argument("--show-config", action="store_true", help="Print current config then exit")
    return p


def main():
    args = build_parser().parse_args()

    if args.show_config:
        print(CONFIG.describe())
        return

    overrides = {
        "DUALSYS_CASE": _env(args.case),
        "DUALSYS_DAYS": _env(args.days),
        "DUALSYS_GUI": _env(args.gui),
        "DUALSYS_CONCURRENCY": _env(args.concurrency),
        "DUALSYS_SEED": _env(args.seed),
        "DUALSYS_NO_MEMORY": (None if args.memory is None else ("0" if args.memory else "1")),
        "DUALSYS_BASELINE": _env(args.baseline),
    }
    overrides = {k: v for k, v in overrides.items() if v is not None}

    print("=" * 70)
    print("🚀 Simulation entry point started")
    print("=" * 70)
    print(CONFIG.describe())
    print("⚠️ This entry point does not generate data. If data is missing, run: python generate.py")

    if CONFIG.pipeline.simulate:
        run_stage("Run simulation", "main.py", overrides)

    print("\n" + "=" * 70)
    print("🎉 All done.")
    print("=" * 70)


if __name__ == "__main__":
    main()
