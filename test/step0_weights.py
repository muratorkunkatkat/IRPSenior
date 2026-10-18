"""Step 0 only: SA with equal move weights, then calibrate weights from build success.

Run from the project root:

    python test/step0_weights.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "test") not in sys.path:
    sys.path.insert(0, str(ROOT / "test"))

from heuristic.search_problem import (
    MOVE_NAMES,
    Solution,
    build_initial_trips,
    calibrated_weights,
    clone_trips,
    evaluate,
    move_stats,
    reset_move_stats,
    set_move_weights,
)
from heuristic.simulated_annealing import paper_temperatures, run_simulated_annealing
from calibrate_metaheuristics import (
    improvement,
    load_instance,
    print_cost,
    save_move_acceptance,
    save_sa_progress,
)

TIME_LIMIT = 150
SEED = 44
ITERATIONS = 1_000_000
T0 = 15.0


def main():
    data, path = load_instance()
    base = build_initial_trips(data)
    initial, _aux = evaluate(Solution(data, clone_trips(base)))
    temps = paper_temperatures(initial.total)
    ratio = temps["T_final"] / temps["T25"]
    t_final = T0 * ratio

    print("Instance:", path)
    print_cost("Starting plan", initial)
    print()
    print(f"STEP 0  SA T0={T0:,.2f}  T_final={t_final:,.4f}  "
          f"(paper ratio {ratio:.6f})  clock cooling over {TIME_LIMIT}s  equal weights")
    print()

    set_move_weights([1.0] * len(MOVE_NAMES))
    reset_move_stats()
    sa0 = run_simulated_annealing(
        data,
        iterations=ITERATIONS,
        seed=SEED,
        trips=base,
        t0=T0,
        t_final=t_final,
        time_limit_sec=TIME_LIMIT,
        log_every=1500,
    )
    print_cost("Step 0 best", sa0["best_cost"])
    print(f"  Improvement {improvement(initial.total, sa0['best_cost'].total):.2f}%")
    print(f"  Iterations {sa0['iterations_run']}  last best {sa0['best_iteration']}")
    print()

    stats = move_stats()
    print(f"{'move':<16} {'attempts':>9} {'built':>9} {'build%':>8} "
          f"{'proposed':>9} {'accepted':>9} {'accept%':>8}")
    for kind in MOVE_NAMES:
        attempts = stats[kind]["attempts"]
        built = stats[kind]["built"]
        proposed = sa0["proposed_counts"].get(kind, 0)
        accepted = sa0["move_counts"].get(kind, 0)
        build_pct = 100.0 * built / attempts if attempts else 0.0
        acc_pct = 100.0 * accepted / proposed if proposed else 0.0
        print(f"{kind:<16} {attempts:9d} {built:9d} {build_pct:7.1f}% "
              f"{proposed:9d} {accepted:9d} {acc_pct:7.1f}%")

    weights = calibrated_weights()
    total = sum(weights)
    print()
    print("Calibrated weights (proportional to 1 / build success, capped at 5x):")
    for kind, weight in zip(MOVE_NAMES, weights):
        print(f"  {kind:<16} {weight:6.3f}   chance to be drawn first {100 * weight / total:5.1f}%")
    print()
    print("CALIBRATED_WEIGHTS = [" + ", ".join(f"{w:.3f}" for w in weights) + "]")

    out = ROOT / "plots" / "initial"
    save_sa_progress(sa0, out, f"Step 0 SA (T0={T0:g}, equal weights)")
    save_move_acceptance(sa0, stats, out)
    print(f"Plots: {out}")


if __name__ == "__main__":
    main()
