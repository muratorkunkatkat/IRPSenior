"""Equal-weight SA on the random start, then build-rate weights.

T0 is the measured-move T25. Cooling reaches the shared floor at 30 minutes.
Weights are 1 / build success, scaled so the smallest is 1 and capped at 5.

    python test/reweight_t25.py
"""

import json
import math
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "test") not in sys.path:
    sys.path.insert(0, str(ROOT / "test"))

from heuristic.search_problem import (
    MOVE_NAMES,
    build_random_trips,
    calibrated_weights,
    move_stats,
    reset_move_stats,
    set_move_weights,
)
from heuristic.simulated_annealing import run_simulated_annealing
from calibrate_metaheuristics import (
    improvement,
    load_instance,
    print_cost,
    save_move_acceptance,
    save_sa_progress,
)

SEED = 44
RUN_SECONDS = 1800
ITERATIONS = 8_000_000
DELTA = 6685.0
T25 = -DELTA / math.log(0.25)
T_FLOOR = -(DELTA / 10.0) / math.log(0.0001)


def main():
    data, path = load_instance()
    trips = build_random_trips(data, random.Random(SEED))
    print("Instance:", path)
    print(f"Random start, seed {SEED}. Equal weights. "
          f"T0={T25:,.2f}  floor={T_FLOOR:,.2f}  {RUN_SECONDS}s")

    set_move_weights([1.0] * len(MOVE_NAMES))
    reset_move_stats()
    run = run_simulated_annealing(
        data,
        iterations=ITERATIONS,
        seed=SEED,
        trips=trips,
        t0=T25,
        t_final=T_FLOOR,
        time_limit_sec=RUN_SECONDS,
        log_every=20_000,
        alpha=None,
        history_every=25,
    )
    print_cost("Best", run["best_cost"])
    print(f"  Improvement {improvement(run['initial_cost'].total, run['best_cost'].total):.2f}%")
    print(f"  Iterations {run['iterations_run']}  best at {run['best_iteration']}")

    stats = move_stats()
    print(f"{'move':<16} {'attempts':>9} {'built':>9} {'build%':>8} "
          f"{'proposed':>9} {'accepted':>9} {'accept%':>8}")
    for kind in MOVE_NAMES:
        attempts = stats[kind]["attempts"]
        built = stats[kind]["built"]
        proposed = run["proposed_counts"].get(kind, 0)
        accepted = run["move_counts"].get(kind, 0)
        build_pct = 100.0 * built / attempts if attempts else 0.0
        acc_pct = 100.0 * accepted / proposed if proposed else 0.0
        print(f"{kind:<16} {attempts:9d} {built:9d} {build_pct:7.1f}% "
              f"{proposed:9d} {accepted:9d} {acc_pct:7.1f}%")

    weights = calibrated_weights()
    total = sum(weights)
    print("Calibrated weights (proportional to 1 / build success, capped at 5x):")
    for kind, weight in zip(MOVE_NAMES, weights):
        print(f"  {kind:<16} {weight:6.3f}   draw {100 * weight / total:5.1f}%")
    print("WEIGHTS = [" + ", ".join(f"{w:.3f}" for w in weights) + "]")

    out = ROOT / "plots" / "initial"
    out.mkdir(parents=True, exist_ok=True)
    save_sa_progress(run, out, f"Random-start SA, T25={T25:,.0f}, equal weights, 30 min")
    save_move_acceptance(run, stats, out)
    payload = {
        "seed": SEED,
        "start": "random",
        "T0": T25,
        "floor": T_FLOOR,
        "seconds": RUN_SECONDS,
        "moves": list(MOVE_NAMES),
        "weights": weights,
        "stats": stats,
        "move_counts": run.get("move_counts", {}),
        "proposed_counts": run.get("proposed_counts", {}),
        "best": run["best_cost"].total,
        "initial": run["initial_cost"].total,
    }
    with open(out / "weights.json", "w") as handle:
        json.dump(payload, handle, indent=2)
    print(f"Plots: {out}")


if __name__ == "__main__":
    main()
