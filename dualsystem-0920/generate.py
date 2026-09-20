# -*- coding: utf-8 -*-
"""Data preparation entry point: generates pricing XML, agents, and NPCs.

Run `python run.py` afterwards to simulate.
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
    p = argparse.ArgumentParser(description="Data generation entry point (pricing / agents / NPCs)")
    p.add_argument("--skip-prices", action="store_true", help="Skip pricing generation")
    p.add_argument("--skip-agents", action="store_true", help="Skip agent generation")
    p.add_argument("--skip-npcs", action="store_true", help="Skip NPC generation")

    p.add_argument("--case", type=str, default=None, help="Road network case (nanjing / 12nodes), overrides config.paths.case_name")
    p.add_argument("--npcs", type=int, default=None, help="NPC count (overrides config.generation.npcs_count)")
    p.add_argument("--seed", type=int, default=None, help="Random seed (agents & NPCs)")
    p.add_argument("--instances", type=int, default=None, help="Instances per archetype")
    p.add_argument("--force-new", action="store_true", help="Force LLM regeneration of archetypes")

    p.add_argument("--show-config", action="store_true", help="Print current config then exit")
    return p


def main():
    args = build_parser().parse_args()

    if args.show_config:
        print(CONFIG.describe())
        return

    overrides = {
        "DUALSYS_CASE": _env(args.case),
        "DUALSYS_NPCS": _env(args.npcs),
        "DUALSYS_SEED": _env(args.seed),
        "DUALSYS_INSTANCES": _env(args.instances),
        "DUALSYS_FORCE_NEW": "1" if args.force_new else None,
    }
    overrides = {k: v for k, v in overrides.items() if v is not None}

    do_prices = CONFIG.pipeline.generate_prices
    do_agents = CONFIG.pipeline.generate_agents
    do_npcs = CONFIG.pipeline.generate_npcs
    if args.skip_prices:
        do_prices = False
    if args.skip_agents:
        do_agents = False
    if args.skip_npcs:
        do_npcs = False

    print("=" * 70)
    print("🛠 Data generation entry point started")
    print("=" * 70)
    print(CONFIG.describe())
    print(f"\nStage toggles: pricing={do_prices}  agents={do_agents}  NPCs={do_npcs}")

    if do_prices:
        run_stage("Generate pricing XML", "overwrite_prices.py", overrides)

    if do_agents:
        run_stage("Generate agents (incl. archetypes)", "generate_agents_file.py", overrides)

    if do_npcs:
        run_stage("Generate NPCs", "generate_npcs.py", overrides)

    print("\n" + "=" * 70)
    print("🎉 Data generation done. Next: python run.py")
    print("=" * 70)


if __name__ == "__main__":
    main()
