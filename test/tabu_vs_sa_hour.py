"""Hour-long tabu vs SA, using the winner of each start.

Greedy: clock-cooled T0=15 against tabu size 125 with 35 neighbors.
Random: clock-cooled T0=150 against tabu size 5 with 35 neighbors.
Both use the final offer weights. One hour each, four hours in total.

The 300s and 600s figures in plots/tabu_vs_sa/greedy and random are left
untouched. These plots go to plots/tabu_vs_sa/hour.

    python test/tabu_vs_sa_hour.py
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
    evaluate,
    set_move_weights,
)
from heuristic.simulated_annealing import paper_temperatures, run_simulated_annealing
from heuristic.tabu_search import run_tabu_search
from calibrate_metaheuristics import improvement, load_instance, print_cost
from tabu_vs_sa import save_comparison

WEIGHTS = [5.0, 5.0, 5.0, 2.177729599401344, 5.0, 5.0, 1.0, 5.0]
SEED = 44
HOUR = 3600
OUT = ROOT / "plots" / "tabu_vs_sa" / "hour"
CACHE = OUT / "_cache"


def _log(text):
    print(text, flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "run_log.txt", "a") as handle:
        handle.write(text + "\n")
        handle.flush()


def _slim(run):
    run.pop("best_trips", None)
    run.pop("accepted_flags", None)
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


def _sa(data, trips, t0, t_final):
    return run_simulated_annealing(
        data,
        iterations=5_000_000,
        seed=SEED,
        trips=trips,
        t0=t0,
        t_final=t_final,
        time_limit_sec=HOUR,
        log_every=20_000,
        alpha=None,
        history_every=25,
    )


def _tabu(data, trips, neighbors, tabu_size):
    return run_tabu_search(
        data,
        iterations=1_000_000,
        neighbors=neighbors,
        tabu_size=tabu_size,
        seed=SEED,
        trips=trips,
        time_limit_sec=HOUR,
        log_every=2_000,
    )


def _one(key, builder):
    run = _load(key)
    if run is None:
        run = _slim(builder())
        _store(key, run)
    else:
        _log(f"Cached {key}")
    return run


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data, path = load_instance()
    set_move_weights(WEIGHTS)
    greedy = build_initial_trips(data)
    greedy_cost, _aux = evaluate(Solution(data, clone_trips(greedy)))
    temps = paper_temperatures(greedy_cost.total)
    ratio = temps["T_final"] / temps["T25"]
    random_trips = build_random_trips(data, random.Random(SEED))
    random_cost, _aux = evaluate(Solution(data, clone_trips(random_trips)))

    _log(f"Instance: {path}")
    _log(f"Weights: {WEIGHTS}")
    _log(f"Clock ratio T_final/T25 = {ratio:.6f}")
    print_cost("Greedy start", greedy_cost)
    print_cost("Random start", random_cost)

    jobs = (
        {
            "name": "greedy",
            "baseline": "the dynamic greedy",
            "trips": greedy,
            "initial": greedy_cost,
            "t0": 15.0,
            "neighbors": 35,
            "tabu_size": 125,
        },
        {
            "name": "random",
            "baseline": "the random plan",
            "trips": random_trips,
            "initial": random_cost,
            "t0": 150.0,
            "neighbors": 35,
            "tabu_size": 5,
        },
    )
    for job in jobs:
        t_final = job["t0"] * ratio
        _log("")
        _log("#" * 60)
        _log(
            f"{job['name']}  SA T0={job['t0']:.0f} floor={t_final:.4f} clock  |  "
            f"tabu n={job['neighbors']} size={job['tabu_size']}"
        )
        _log("#" * 60)
        sa = _one(
            f"{job['name']}_sa",
            lambda job=job, t_final=t_final: _sa(data, job["trips"], job["t0"], t_final),
        )
        print_cost(f"{job['name']} SA best", sa["best_cost"])
        _log(
            f"  SA improvement {improvement(job['initial'].total, sa['best_cost'].total):.2f}%  "
            f"best iter {sa['best_iteration']} / {sa['iterations_run']}  "
            f"end T={sa['history'][-1]['temperature']:,.2f}"
        )
        tabu = _one(
            f"{job['name']}_tabu",
            lambda job=job: _tabu(data, job["trips"], job["neighbors"], job["tabu_size"]),
        )
        print_cost(f"{job['name']} tabu best", tabu["best_cost"])
        _log(
            f"  Tabu improvement {improvement(job['initial'].total, tabu['best_cost'].total):.2f}%  "
            f"best iter {tabu['best_iteration']} / {tabu['iterations_run']}  "
            f"forbidden {tabu['forbidden_share']:.3f}%  "
            f"worse steps {tabu['worse_steps']}  all-tabu {tabu['all_tabu_iterations']}"
        )
        gap = sa["best_cost"].total - tabu["best_cost"].total
        lower = "tabu" if gap > 0 else "SA"
        _log(f"  {lower} lower by {abs(gap):,.2f}")
        out_dir = OUT / job["name"]
        save_comparison(sa, tabu, out_dir, job["baseline"])
        _log(f"Plots: {out_dir}")

    _log("Finished.")


if __name__ == "__main__":
    main()
