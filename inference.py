"""Baseline inference script for the SuperStar Sprint Environment.

Runs deterministic sprint simulations with reproducible
[START]/[STEP]/[END] logs for each episode.
"""

from __future__ import annotations

import os
import sys

# Ensure src/ and server/ are importable
_root = os.path.dirname(os.path.abspath(__file__))
for _sub in ("src", "server"):
    _p = os.path.join(_root, _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)

from server.superstar import SprintEnv, SprintAction, run_simulation
from src.scenarios import BUILT_IN_SCENARIOS, get_default_scenario


def main() -> None:
    print("[START] env=SuperStar scenario=default")

    result = run_simulation(scenario=get_default_scenario(), verbose=True)
    total_reward = result["total_reward"]
    state = result["final_state"]

    tasks_done = sum(
        1 for t in state.get("backlog", []) if t["status"] == "done"
    )
    total_tasks = len(state.get("backlog", []))

    print(f"\n[END] success=true total_reward={total_reward:.3f} "
          f"tasks_done={tasks_done}/{total_tasks}")


if __name__ == "__main__":
    main()
