"""Round 2: paper-alpha SA (T75/T50/T25) and tabu size 5/12/25 at neighbors=16.

Run from the project root:

    python test/calibrate_round2.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# stdlib `test` shadows the project folder, so load helpers from this file's directory.
if str(ROOT / "test") not in sys.path:
    sys.path.insert(0, str(ROOT / "test"))

from heuristic.search_problem import (
    Solution,
    build_initial_trips,
    clone_trips,
    count_active,
    evaluate,
    reset_move_stats,
    set_move_weights,
)
from heuristic.simulated_annealing import paper_temperatures, run_simulated_annealing
from heuristic.tabu_search import run_tabu_search
from calibrate_metaheuristics import (
    SA_VARIANT_COLORS,
    TABU_VARIANT_COLORS,
    TIME_LIMIT,
    ITERATIONS,
    SEED,
    improvement,
    load_instance,
    print_cost,
    save_sa_progress,
    save_tabu_diagnostics,
    save_variant_comparison,
)

# Step 0 weights: 1 / build-success, capped at 5x.
CALIBRATED_WEIGHTS = [5.0, 5.0, 5.0, 2.312, 5.0, 5.0, 1.0, 3.754]

# Paper geometric cooling: T <- alpha * T after every L scored neighbors.
PAPER_ALPHA = 0.995
PAPER_L = 10


def main():
    data, path = load_instance()
    set_move_weights(CALIBRATED_WEIGHTS)
    base = build_initial_trips(data)
    initial, _aux = evaluate(Solution(data, clone_trips(base)))
    temps = paper_temperatures(initial.total)

    print("Instance:", path)
    print(f"Dealers {len(data['nodes_no_factory'])} | Trucks {len(data['K'])} | "
          f"Days {len(data['T'])} | greedy trips {count_active(base)}")
    print_cost("Starting plan", initial)
    print()
    print(f"Paper T0  T75={temps['T75']:,.2f}  T50={temps['T50']:,.2f}  "
          f"T25={temps['T25']:,.2f}  T_final={temps['T_final']:,.2f}")
    print(f"Paper cooling  alpha={PAPER_ALPHA}  L={PAPER_L}  floor at T_final")
    print(f"Tabu neighbors frozen at 16; sizes 5, 12, 25")
    print(f"Time limit {TIME_LIMIT}s | seed {SEED}")
    print()

    sa_runs = []
    for label, key in (("T75", "T75"), ("T50", "T50"), ("T25", "T25")):
        reset_move_stats()
        print("=" * 60)
        print(f"SA {label}  alpha={PAPER_ALPHA}  L={PAPER_L}")
        print("=" * 60)
        run = run_simulated_annealing(
            data,
            iterations=ITERATIONS,
            seed=SEED,
            trips=base,
            t0=temps[key],
            t_final=temps["T_final"],
            time_limit_sec=TIME_LIMIT,
            log_every=1500,
            alpha=PAPER_ALPHA,
            cool_every=PAPER_L,
        )
        run["label"] = f"{label} α={PAPER_ALPHA}"
        sa_runs.append(run)
        print_cost(f"{label} best", run["best_cost"])
        print(f"  Improvement {improvement(initial.total, run['best_cost'].total):.2f}%")
        print(f"  Last improvement iter {run['best_iteration']} / {run['iterations_run']}")
        print(f"  End T={run['history'][-1]['temperature']:,.2f}  "
              f"current={run['history'][-1]['current']:,.2f}")
        print(f"  Accepted: {run['move_counts']}")
        print()

    sa_dir = ROOT / "plots" / "sa"
    save_variant_comparison(sa_runs, sa_dir, SA_VARIANT_COLORS, "SA paper alpha")
    save_sa_progress(sa_runs[1], sa_dir, f"SA T50  alpha={PAPER_ALPHA} L={PAPER_L}")
    print(f"SA plots: {sa_dir}")
    best_sa = min(sa_runs, key=lambda run: run["best_cost"].total)
    print(f"Best SA on this clock: {best_sa['label']} ({best_sa['best_cost'].total:,.2f})")
    print()

    tabu_runs = []
    for size in (5, 12, 25):
        reset_move_stats()
        label = f"size={size}"
        print("=" * 60)
        print(f"Tabu neighbors=16  {label}")
        print("=" * 60)
        run = run_tabu_search(
            data,
            iterations=ITERATIONS,
            neighbors=16,
            tabu_size=size,
            seed=SEED,
            trips=base,
            time_limit_sec=TIME_LIMIT,
            log_every=25,
        )
        run["label"] = label
        tabu_runs.append(run)
        print_cost(f"{label} best", run["best_cost"])
        print(f"  Improvement {improvement(initial.total, run['best_cost'].total):.2f}%")
        print(f"  Iterations {run['iterations_run']}  last best {run['best_iteration']}")
        print(f"  Forbidden share {run['forbidden_share']:.3f}%  "
              f"({run['forbidden_candidates']}/{run['scored_candidates']})")
        print(f"  All-tabu {run['all_tabu_iterations']}  worse steps {run['worse_steps']}")
        print(f"  Accepted: {run['move_counts']}")
        print()

    tabu_dir = ROOT / "plots" / "tabu" / "size"
    save_variant_comparison(tabu_runs, tabu_dir, TABU_VARIANT_COLORS, "Tabu size")
    save_tabu_diagnostics(tabu_runs, tabu_dir)
    print(f"Tabu size plots: {tabu_dir}")


if __name__ == "__main__":
    main()
