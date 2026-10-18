"""Five SA starts on a 10x temperature ladder. Clock cooling, no paper alpha.

Run from the project root:

    python test/sa_t0_ladder.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "test") not in sys.path:
    sys.path.insert(0, str(ROOT / "test"))

from heuristic.search_problem import (
    Solution,
    build_initial_trips,
    clone_trips,
    evaluate,
    reset_move_stats,
    set_move_weights,
)
from heuristic.simulated_annealing import paper_temperatures, run_simulated_annealing
from calibrate_metaheuristics import (
    improvement,
    load_instance,
    print_cost,
    save_sa_progress,
    save_variant_comparison,
)

TIME_LIMIT = 120
SEED = 44
ITERATIONS = 1_000_000
CALIBRATED_WEIGHTS = [5.0, 5.0, 5.0, 2.312, 5.0, 5.0, 1.0, 3.754]
LADDER_T0 = [None, 15_000.0, 1_500.0, 150.0, 15.0]  # None = paper T25
COLORS = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7"]


def main():
    data, path = load_instance()
    set_move_weights(CALIBRATED_WEIGHTS)
    base = build_initial_trips(data)
    initial, _aux = evaluate(Solution(data, clone_trips(base)))
    temps = paper_temperatures(initial.total)
    paper_t25 = temps["T25"]
    paper_tf = temps["T_final"]
    ratio = paper_tf / paper_t25

    starts = []
    for raw in LADDER_T0:
        t0 = paper_t25 if raw is None else raw
        starts.append((t0, t0 * ratio))

    labels = [
        f"T25 ({starts[0][0]:,.0f})",
        f"1e4 ({starts[1][0]:,.0f})",
        f"1e3 ({starts[2][0]:,.0f})",
        f"1e2 ({starts[3][0]:,.0f})",
        f"1e1 ({starts[4][0]:,.0f})",
    ]

    print("Instance:", path)
    print_cost("Starting plan", initial)
    print()
    print(f"Paper T25={paper_t25:,.2f}  T_final={paper_tf:,.2f}  "
          f"ratio T_final/T0={ratio:.6f}")
    print(f"Clock cooling over {TIME_LIMIT}s | seed {SEED} | no paper alpha")
    print(f"{'label':<22} {'T0':>12} {'T_final':>12}")
    for label, (t0, tf) in zip(labels, starts):
        print(f"{label:<22} {t0:12,.2f} {tf:12,.2f}")
    print()

    sa_dir = ROOT / "plots" / "sa"
    runs = []
    for label, (t0, tf) in zip(labels, starts):
        reset_move_stats()
        print("=" * 60)
        print(f"SA {label}")
        print("=" * 60)
        run = run_simulated_annealing(
            data,
            iterations=ITERATIONS,
            seed=SEED,
            trips=base,
            t0=t0,
            t_final=tf,
            time_limit_sec=TIME_LIMIT,
            log_every=1500,
            alpha=None,
        )
        run["label"] = label
        runs.append(run)
        print_cost(f"{label} best", run["best_cost"])
        print(f"  Improvement {improvement(initial.total, run['best_cost'].total):.2f}%")
        print(f"  Last improvement iter {run['best_iteration']} / {run['iterations_run']}")
        print(f"  End T={run['history'][-1]['temperature']:,.4f}  "
              f"current={run['history'][-1]['current']:,.2f}")
        print(f"  Accepted: {run['move_counts']}")
        print()
        slug = label.split()[0].replace("(", "").replace(")", "")
        save_sa_progress(run, sa_dir / slug, f"SA {label}")

    save_variant_comparison(runs, sa_dir, COLORS, "SA T0 ladder")
    print(f"Comparison plots: {sa_dir}")
    best = min(runs, key=lambda run: run["best_cost"].total)
    print(f"Best on this clock: {best['label']} ({best['best_cost'].total:,.2f})")


if __name__ == "__main__":
    main()
