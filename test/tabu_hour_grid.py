"""One-hour tabu grid with the final offer weights.

Nine settings on the greedy start and the same nine on the random start:
tabu size 5, 25, and 125, each with 8, 16, and 35 neighbors. One hour each.

Plots land in plots/tabu/greedy/size_<n> and plots/tabu/random/size_<n>.
Each size folder compares its three neighborhood sizes.

    python test/tabu_hour_grid.py
"""

import pickle
import random
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
    evaluate,
    set_move_weights,
)
from heuristic.tabu_search import run_tabu_search
from calibrate_metaheuristics import (
    TABU_VARIANT_COLORS,
    improvement,
    load_instance,
    print_cost,
    save_tabu_diagnostics,
    save_variant_comparison,
)

# One hour of equal-weight SA at clock-cooled T0=15. 1 / build success, cap 5.
WEIGHTS = [5.0, 5.0, 5.0, 2.177729599401344, 5.0, 5.0, 1.0, 5.0]
SEED = 44
HOUR = 3600
ITERATIONS = 1_000_000
SIZES = (5, 25, 125)
NEIGHBORS = (8, 16, 35)
OUT = ROOT / "plots" / "tabu"
CACHE = OUT / "_cache"


def _log(text):
    print(text, flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "hour_log.txt", "a") as handle:
        handle.write(text + "\n")
        handle.flush()


def _slim(run):
    run.pop("best_trips", None)
    return run


def _load(key):
    path = CACHE / f"{key}.pkl"
    if not path.exists():
        return None
    with open(path, "rb") as handle:
        return pickle.load(handle)


def _store(key, run):
    CACHE.mkdir(parents=True, exist_ok=True)
    with open(CACHE / f"{key}.pkl", "wb") as handle:
        pickle.dump(run, handle, protocol=pickle.HIGHEST_PROTOCOL)


def _series(history, key):
    return [row[key] for row in history]


def _save_run(run, out_dir, title):
    out_dir.mkdir(parents=True, exist_ok=True)
    history = run["history"]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(_series(history, "seconds"), _series(history, "current"),
            color="#F4A261", linewidth=1, label="Current")
    ax.plot(_series(history, "seconds"), _series(history, "best"),
            color="#D55E00", linewidth=2, label="Best")
    ax.set_title(title)
    ax.set_xlabel("Seconds")
    ax.set_ylabel("Objective value")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "convergence.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(_series(history, "seconds"), _series(history, "best_transport"),
            color="#0072B2", linewidth=2, label="Transport")
    ax.plot(_series(history, "seconds"), _series(history, "best_backorder"),
            color="#D55E00", linewidth=2, label="Backorder")
    ax.plot(_series(history, "seconds"), _series(history, "best_holding"),
            color="#009E73", linewidth=2, label="Holding")
    ax.plot(_series(history, "seconds"), _series(history, "best_unsold"),
            color="#CC79A7", linewidth=2, label="Unsold")
    ax.set_title(f"{title} — best cost pieces")
    ax.set_xlabel("Seconds")
    ax.set_ylabel("Cost")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "components_vs_time.png", dpi=140)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data, path = load_instance()
    set_move_weights(WEIGHTS)
    greedy = build_initial_trips(data)
    greedy_cost, _aux = evaluate(Solution(data, clone_trips(greedy)))
    random_trips = build_random_trips(data, random.Random(SEED))
    random_cost, _aux = evaluate(Solution(data, clone_trips(random_trips)))

    _log(f"Instance: {path}")
    _log(f"Weights: {WEIGHTS}")
    print_cost("Greedy start", greedy_cost)
    print_cost("Random start", random_cost)
    _log(f"Sizes {list(SIZES)}  neighbors {list(NEIGHBORS)}  {HOUR:.0f}s each")

    for start_name, trips, initial, baseline in (
        ("greedy", greedy, greedy_cost, "Greedy"),
        ("random", random_trips, random_cost, "Random"),
    ):
        _log("")
        _log("#" * 60)
        _log(f"{start_name}  baseline {initial.total:,.2f}")
        _log("#" * 60)
        for size in SIZES:
            runs = []
            out_dir = OUT / start_name / f"size_{size}"
            for neighbors in NEIGHBORS:
                label = f"n={neighbors}"
                key = f"{start_name}_size{size}_n{neighbors}"
                _log("")
                _log("=" * 60)
                _log(f"{start_name}  size={size}  {label}")
                _log("=" * 60)
                run = _load(key)
                if run is None:
                    run = _slim(run_tabu_search(
                        data,
                        iterations=ITERATIONS,
                        neighbors=neighbors,
                        tabu_size=size,
                        seed=SEED,
                        trips=trips,
                        time_limit_sec=HOUR,
                        log_every=2_000,
                    ))
                    _store(key, run)
                else:
                    _log(f"Cached {key}")
                run["label"] = label
                runs.append(run)
                print_cost(f"{start_name} size={size} {label} best", run["best_cost"])
                _log(
                    f"  Improvement {improvement(initial.total, run['best_cost'].total):.2f}%  "
                    f"best iter {run['best_iteration']} / {run['iterations_run']}  "
                    f"forbidden {run['forbidden_share']:.3f}%  "
                    f"worse steps {run['worse_steps']}  "
                    f"all-tabu {run['all_tabu_iterations']}"
                )
                _save_run(run, out_dir / f"n{neighbors}", f"Tabu size {size}, {label}")
            save_variant_comparison(
                runs, out_dir, TABU_VARIANT_COLORS,
                f"Tabu {start_name} size {size}", baseline,
            )
            save_tabu_diagnostics(runs, out_dir)
            best = min(runs, key=lambda item: item["best_cost"].total)
            _log(f"Best {start_name} size {size}: {best['label']} "
                 f"({best['best_cost'].total:,.2f})")
            _log(f"Plots: {out_dir}")
        _log("")

    _log("Finished.")


if __name__ == "__main__":
    main()
