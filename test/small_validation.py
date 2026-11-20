"""Greedy heuristic and metaheuristics on the small exact-model instances (10 dealers, 7 days, 2 car types).

    python test/small_validation.py            # greedy table (4, 6, 8 trucks)
    python test/small_validation.py meta       # SA 500, SA 50 and TS 70/625, 120 s each, 9 runs in parallel
    python test/small_validation.py job 4 500  # one metaheuristic run
"""

import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "test"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from data.generator import generate_network_data
from heuristic.search_problem import Solution, build_initial_trips, clone_trips, evaluate, set_move_weights
from heuristic.simulated_annealing import run_simulated_annealing
from heuristic.tabu_search import run_tabu_search

TRUCKS = (4, 6, 8)
ALGOS = ("500", "50", "tabu")
SECONDS = 120


def small_instance(trucks):
    data = generate_network_data(num_dealers=10, num_trucks=trucks, days=7, num_car_types=2)
    n_nodes, n_types, n_days = len(data["N"]), len(data["C"]), len(data["T"])
    A = [[0] * n_nodes for _ in range(n_nodes)]
    R = [[0.0] * n_nodes for _ in range(n_nodes)]
    for (i, j), value in data["A"].items():
        A[i][j] = value
    for (i, j), value in data["R"].items():
        R[i][j] = value
    F = [[[0] * n_days for _ in range(n_types)] for _ in range(n_nodes)]
    for (i, c, t), value in data["F"].items():
        F[i][c][t] = value
    data["A"], data["R"], data["F"] = A, R, F
    for key in ("V", "B", "H", "G", "Q", "P"):
        value = data.get(key)
        if isinstance(value, dict):
            data[key] = [value.get(i, 0) for i in range(max(value) + 1)]
    return data


def greedy_table():
    print(f"{'Trucks':<7}{'Obj. Value':>12}{'Trans. Cost':>13}{'Backorder':>11}{'Holding':>9}{'Unsold':>8}{'Time (s)':>10}")
    for trucks in TRUCKS:
        data = small_instance(trucks)
        started = time.perf_counter()
        trips = build_initial_trips(data)
        seconds = time.perf_counter() - started
        cost, _aux = evaluate(Solution(data, clone_trips(trips)))
        print(f"{trucks:<7}{cost.total:>12.2f}{cost.transport:>13.2f}{cost.backorder:>11.2f}"
              f"{cost.holding:>9.2f}{cost.unsold:>8.2f}{seconds:>10.4f}")


def job(trucks, algo):
    weights = json.loads((ROOT / "plots" / "analysis" / "weights.json").read_text())["weights"]
    set_move_weights(weights)
    data = small_instance(trucks)
    trips = build_initial_trips(data)
    if algo == "tabu":
        run = run_tabu_search(data, iterations=10 ** 7, neighbors=70, tabu_size=625, seed=44, trips=trips,
                              time_limit_sec=SECONDS)
    else:
        t0 = float(algo)
        run = run_simulated_annealing(data, iterations=10 ** 9, seed=44, trips=trips, t0=t0,
                                      t_final=t0 * 72.58 / 4822.2, time_limit_sec=SECONDS, alpha=None)
    best = run["best_cost"]
    print(json.dumps({"trucks": trucks, "algo": algo, "total": best.total, "transport": best.transport,
                      "holding": best.holding, "backorder": best.backorder, "unsold": best.unsold}))


def meta():
    procs = [(t, a, subprocess.Popen([sys.executable, __file__, "job", str(t), a], stdout=subprocess.PIPE,
                                     stderr=subprocess.DEVNULL, text=True))
             for t in TRUCKS for a in ALGOS]
    rows = []
    for trucks, algo, proc in procs:
        out, _err = proc.communicate()
        line = out.strip().splitlines()[-1]
        rows.append(json.loads(line))
        print(line)
    (ROOT / "plots" / "report" / "small_validation.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "meta":
        meta()
    elif len(sys.argv) >= 2 and sys.argv[1] == "job":
        job(int(sys.argv[2]), sys.argv[3])
    else:
        greedy_table()
