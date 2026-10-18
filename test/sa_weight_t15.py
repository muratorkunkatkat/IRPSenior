"""Equal-weight SA at T0=1500, then offer-weights from build success.

Does not rerun the temperature ladder or the tabu tests.

    python test/sa_weight_t15.py
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
    count_active,
    current_move_weights,
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
T0 = 1500.0


def main():
    data, path = load_instance()
    set_move_weights([1.0] * len(MOVE_NAMES))
    reset_move_stats()
    base = build_initial_trips(data)
    initial, _aux = evaluate(Solution(data, clone_trips(base)))
    temps = paper_temperatures(initial.total)
    ratio = temps["T_final"] / temps["T25"]
    t_final = T0 * ratio

    print("Instance:", path)
    print(f"Dynamic greedy | active trips {count_active(base)}")
    print_cost("Starting plan", initial)
    print()
    print(f"Equal weights. T0={T0:.2f}  T_final={t_final:.4f}  "
          f"(same ratio as paper T25/T_final = {ratio:.6f})")
    print(f"Clock cooling over {TIME_LIMIT}s | seed {SEED}")
    print()

    sa0 = run_simulated_annealing(
        data,
        iterations=ITERATIONS,
        seed=SEED,
        trips=base,
        t0=T0,
        t_final=t_final,
        time_limit_sec=TIME_LIMIT,
        log_every=2000,
        alpha=None,
    )
    print_cost("T0=1500 equal-weight best", sa0["best_cost"])
    print(f"  Improvement {improvement(initial.total, sa0['best_cost'].total):.2f}%")
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
    print()
    print("New offer weights (1 / build success, capped at 5x, scaled so the smallest is 1):")
    for kind, weight in zip(MOVE_NAMES, weights):
        print(f"  {kind:<16} {weight:.3f}")
    print()
    print("Not written into the later SA/tabu scripts. Confirm these rates first.")

    out = ROOT / "plots" / "initial"
    save_sa_progress(sa0, out, "Equal-weight SA at T0=1500")
    save_move_acceptance(sa0, stats, out)
    print(f"Plots: {out}")
    # keep the function import used so a reader sees weights are computed, not stored
    _ = current_move_weights


if __name__ == "__main__":
    main()
