"""Re-run the SA ladder and the tabu sweeps from a greedy start and a random start.

Greedy runs last 100s. Random-start runs last 200s. The winning SA temperature
and the winning tabu setting of each start are then compared for 300s.

    python test/rerun_starts.py
"""

import pickle
import random
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
    build_random_trips,
    clone_trips,
    count_active,
    evaluate,
    set_move_weights,
)
from heuristic.simulated_annealing import paper_temperatures, run_simulated_annealing
from heuristic.tabu_search import run_tabu_search
from calibrate_metaheuristics import (
    TABU_VARIANT_COLORS,
    improvement,
    load_instance,
    print_cost,
    save_sa_progress,
    save_tabu_diagnostics,
    save_variant_comparison,
)
from tabu_vs_sa import save_comparison

# Step 0 at T0=1500, equal weights: 1 / build success, cap 5, smallest = 1.
CALIBRATED_WEIGHTS = [5.0, 5.0, 5.0, 2.312, 5.0, 5.0, 1.0, 3.754]
SEED = 44
ITERATIONS = 1_000_000
GREEDY_LIMIT = 100
RANDOM_LIMIT = 200
COMPARE_LIMIT = 300
NEIGHBOR_SWEEP = (16, 35, 71)
SIZE_SWEEP = (5, 12, 25)
NEIGHBOR_SWEEP_SIZE = 12
SA_COLORS = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7"]
CACHE = Path("/tmp/irp_rerun_s1s2")


def _slug(label):
    return "".join(ch if ch.isalnum() else "_" for ch in label).strip("_")


def _load_or_run(key, builder):
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"{_slug(key)}.pkl"
    if path.exists():
        print(f"Cached {key}", flush=True)
        with open(path, "rb") as handle:
            return pickle.load(handle)
    run = builder()
    with open(path, "wb") as handle:
        pickle.dump(run, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return run


def _report(label, initial, run):
    print_cost(f"{label} best", run["best_cost"])
    print(f"  Improvement {improvement(initial.total, run['best_cost'].total):.2f}%")
    print(f"  Last improvement iter {run['best_iteration']} / {run['iterations_run']}")
    print(f"  Accepted: {run['move_counts']}")
    print(flush=True)


def sa_ladder(data, base, initial, temps, limit, out_dir, baseline_name):
    paper_t25 = temps["T25"]
    paper_tf = temps["T_final"]
    ratio = paper_tf / paper_t25
    raw_starts = [None, 15_000.0, 1_500.0, 150.0, 15.0]
    starts = []
    for raw in raw_starts:
        t0 = paper_t25 if raw is None else raw
        starts.append((t0, t0 * ratio))
    labels = [
        f"T25 ({starts[0][0]:,.0f})",
        f"1e4 ({starts[1][0]:,.0f})",
        f"1e3 ({starts[2][0]:,.0f})",
        f"1e2 ({starts[3][0]:,.0f})",
        f"1e1 ({starts[4][0]:,.0f})",
    ]
    print(f"SA ladder | {baseline_name} | {limit:.0f}s | ratio {ratio:.6f}")
    for label, (t0, tf) in zip(labels, starts):
        print(f"  {label:<22} T0={t0:12,.2f}  T_final={tf:12,.2f}")
    print(flush=True)

    runs = []
    for label, (t0, tf) in zip(labels, starts):
        print("=" * 60)
        print(f"SA {baseline_name} {label}  {limit:.0f}s")
        print("=" * 60, flush=True)

        def build(label=label, t0=t0, tf=tf):
            return run_simulated_annealing(
                data,
                iterations=ITERATIONS,
                seed=SEED,
                trips=base,
                t0=t0,
                t_final=tf,
                time_limit_sec=limit,
                log_every=2000,
                alpha=None,
            )

        run = _load_or_run(f"sa_{baseline_name}_{label}_{limit}", build)
        run["label"] = label
        runs.append(run)
        _report(f"{baseline_name} {label}", initial, run)
        save_sa_progress(run, out_dir / label.split()[0], f"SA {label} from {baseline_name}")

    save_variant_comparison(runs, out_dir, SA_COLORS, f"SA from {baseline_name}", baseline_name)
    best = min(runs, key=lambda item: item["best_cost"].total)
    print(f"Best SA from {baseline_name}: {best['label']} ({best['best_cost'].total:,.2f})")
    print(f"Plots: {out_dir}", flush=True)
    return best, starts, labels, runs


def tabu_neighbors(data, base, initial, limit, out_dir, baseline_name):
    runs = []
    for neighbors in NEIGHBOR_SWEEP:
        label = f"n={neighbors}"
        print("=" * 60)
        print(f"Tabu {baseline_name} {label}  size={NEIGHBOR_SWEEP_SIZE}  {limit:.0f}s")
        print("=" * 60, flush=True)

        def build(neighbors=neighbors):
            return run_tabu_search(
                data,
                iterations=ITERATIONS,
                neighbors=neighbors,
                tabu_size=NEIGHBOR_SWEEP_SIZE,
                seed=SEED,
                trips=base,
                time_limit_sec=limit,
                log_every=50,
            )

        run = _load_or_run(
            f"tabu_{baseline_name}_{label}_size{NEIGHBOR_SWEEP_SIZE}_{limit}",
            build,
        )
        run["label"] = label
        runs.append(run)
        _report(f"{baseline_name} {label}", initial, run)
        print(f"  Forbidden {run['forbidden_share']:.3f}%  "
              f"worse steps {run['worse_steps']}  all-tabu {run['all_tabu_iterations']}")

    save_variant_comparison(
        runs, out_dir, TABU_VARIANT_COLORS, f"Tabu neighbors from {baseline_name}", baseline_name,
    )
    save_tabu_diagnostics(runs, out_dir)
    best = min(runs, key=lambda item: item["best_cost"].total)
    chosen = best["parameters"]["neighbors"]
    print(f"Best neighborhood from {baseline_name}: {chosen} "
          f"({best['best_cost'].total:,.2f})")
    print(f"Plots: {out_dir}", flush=True)
    return chosen


def tabu_sizes(data, base, initial, neighbors, limit, out_dir, baseline_name):
    runs = []
    for size in SIZE_SWEEP:
        label = f"size={size}"
        print("=" * 60)
        print(f"Tabu {baseline_name} neighbors={neighbors}  {label}  {limit:.0f}s")
        print("=" * 60, flush=True)

        def build(size=size):
            return run_tabu_search(
                data,
                iterations=ITERATIONS,
                neighbors=neighbors,
                tabu_size=size,
                seed=SEED,
                trips=base,
                time_limit_sec=limit,
                log_every=50,
            )

        run = _load_or_run(
            f"tabu_{baseline_name}_n{neighbors}_{label}_{limit}",
            build,
        )
        run["label"] = label
        runs.append(run)
        _report(f"{baseline_name} {label}", initial, run)
        print(f"  Forbidden {run['forbidden_share']:.3f}%  "
              f"worse steps {run['worse_steps']}  all-tabu {run['all_tabu_iterations']}")

    save_variant_comparison(
        runs, out_dir, TABU_VARIANT_COLORS,
        f"Tabu size from {baseline_name} (n={neighbors})", baseline_name,
    )
    save_tabu_diagnostics(runs, out_dir)
    best = min(runs, key=lambda item: item["best_cost"].total)
    chosen = best["parameters"]["tabu_size"]
    print(f"Best tabu size from {baseline_name}: {chosen} "
          f"({best['best_cost'].total:,.2f})")
    print(f"Plots: {out_dir}", flush=True)
    return chosen, best


def compare(data, base, initial, t0, t_final, neighbors, tabu_size, out_dir, baseline_name):
    print("=" * 60)
    print(f"300s comparison from {baseline_name}")
    print(f"  SA T0={t0:,.2f}  T_final={t_final:,.2f}")
    print(f"  Tabu neighbors={neighbors}  size={tabu_size}")
    print("=" * 60, flush=True)

    def build_sa():
        return run_simulated_annealing(
            data,
            iterations=ITERATIONS,
            seed=SEED,
            trips=base,
            t0=t0,
            t_final=t_final,
            time_limit_sec=COMPARE_LIMIT,
            log_every=2000,
            alpha=None,
        )

    def build_tabu():
        return run_tabu_search(
            data,
            iterations=ITERATIONS,
            neighbors=neighbors,
            tabu_size=tabu_size,
            seed=SEED,
            trips=base,
            time_limit_sec=COMPARE_LIMIT,
            log_every=50,
        )

    sa = _load_or_run(f"cmp_sa_{baseline_name}_{t0}_{COMPARE_LIMIT}", build_sa)
    _report(f"{baseline_name} SA 300s", initial, sa)
    tabu = _load_or_run(
        f"cmp_tabu_{baseline_name}_n{neighbors}_s{tabu_size}_{COMPARE_LIMIT}",
        build_tabu,
    )
    _report(f"{baseline_name} tabu 300s", initial, tabu)
    phrase = "the dynamic greedy" if baseline_name == "Greedy" else "the random start"
    save_comparison(sa, tabu, out_dir, phrase)
    print(f"Comparison plots: {out_dir}", flush=True)
    return sa, tabu


def main():
    data, path = load_instance()
    set_move_weights(CALIBRATED_WEIGHTS)
    greedy = build_initial_trips(data)
    greedy_cost, _aux = evaluate(Solution(data, clone_trips(greedy)))
    random_trips = build_random_trips(data, random.Random(SEED))
    random_cost, _aux = evaluate(Solution(data, clone_trips(random_trips)))
    temps = paper_temperatures(greedy_cost.total)

    print("Instance:", path)
    print(f"Weights: {CALIBRATED_WEIGHTS}")
    print(f"Greedy trips {count_active(greedy)}")
    print_cost("Greedy start", greedy_cost)
    print(f"Random trips {count_active(random_trips)}")
    print_cost("Random start", random_cost)
    print()
    print(f"Paper temperatures kept from the greedy total "
          f"T25={temps['T25']:,.2f}  T_final={temps['T_final']:,.2f}")
    print(f"Greedy calibrations {GREEDY_LIMIT}s | random calibrations {RANDOM_LIMIT}s | "
          f"final comparison {COMPARE_LIMIT}s | seed {SEED}")
    print(flush=True)

    sa_greedy_best, sa_starts, sa_labels, _sa_g = sa_ladder(
        data, greedy, greedy_cost, temps, GREEDY_LIMIT,
        ROOT / "plots" / "sa" / "greedy", "Greedy",
    )
    sa_random_best, _starts_r, _labels_r, _sa_r = sa_ladder(
        data, random_trips, random_cost, temps, RANDOM_LIMIT,
        ROOT / "plots" / "sa" / "random", "Random",
    )

    greedy_neighbors = tabu_neighbors(
        data, greedy, greedy_cost, GREEDY_LIMIT,
        ROOT / "plots" / "tabu" / "greedy", "Greedy",
    )
    _size, greedy_tabu_best = tabu_sizes(
        data, greedy, greedy_cost, greedy_neighbors, GREEDY_LIMIT,
        ROOT / "plots" / "tabu" / "greedy" / "size", "Greedy",
    )
    random_neighbors = tabu_neighbors(
        data, random_trips, random_cost, RANDOM_LIMIT,
        ROOT / "plots" / "tabu" / "random", "Random",
    )
    _size_r, random_tabu_best = tabu_sizes(
        data, random_trips, random_cost, random_neighbors, RANDOM_LIMIT,
        ROOT / "plots" / "tabu" / "random" / "size", "Random",
    )

    greedy_t0 = sa_greedy_best["parameters"]["T0"]
    greedy_tf = sa_greedy_best["parameters"]["T_final"]
    random_t0 = sa_random_best["parameters"]["T0"]
    random_tf = sa_random_best["parameters"]["T_final"]
    compare(
        data, greedy, greedy_cost, greedy_t0, greedy_tf,
        greedy_tabu_best["parameters"]["neighbors"],
        greedy_tabu_best["parameters"]["tabu_size"],
        ROOT / "plots" / "tabu_vs_sa" / "greedy", "Greedy",
    )
    compare(
        data, random_trips, random_cost, random_t0, random_tf,
        random_tabu_best["parameters"]["neighbors"],
        random_tabu_best["parameters"]["tabu_size"],
        ROOT / "plots" / "tabu_vs_sa" / "random", "Random",
    )
    print("All runs finished.")
    print(f"Greedy winner SA {sa_greedy_best['label']} vs tabu "
          f"n={greedy_tabu_best['parameters']['neighbors']} "
          f"size={greedy_tabu_best['parameters']['tabu_size']}")
    print(f"Random winner SA {sa_random_best['label']} vs tabu "
          f"n={random_tabu_best['parameters']['neighbors']} "
          f"size={random_tabu_best['parameters']['tabu_size']}")


if __name__ == "__main__":
    main()
