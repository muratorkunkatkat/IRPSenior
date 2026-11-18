"""1000-minute extensions. Does not touch the 400-minute cache.

    python test/extend_1000.py plots
    python test/extend_1000.py
    python test/extend_1000.py job greedy 00_original sa_T75
"""

import json
import os
import pickle
import signal
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "test") not in sys.path:
    sys.path.insert(0, str(ROOT / "test"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import analysis_long as study
from analysis_grid import build_instance
from calibrate_metaheuristics import improvement, load_instance, print_cost
from heuristic.search_problem import Solution, clone_trips, evaluate
from heuristic.simulated_annealing import run_simulated_annealing

EXTEND_SECONDS = 1000 * 60
SA_ITERATIONS = 80_000_000
OUT = study.OUT / "extended"
CACHE = OUT / "_cache"
LOGS = OUT / "logs"
JOBS = (
    ("greedy", "00_original", "sa_T75"),
    ("greedy", "00_original", "sa_500"),
    ("random", "00_original", "sa_T75"),
    ("greedy", "02_growth_no_fleet", "sa_500"),
    ("greedy", "02_growth_no_fleet", "sa_T25"),
    ("greedy", "02_growth_no_fleet", "sa_T75"),
)


def _log(message):
    LOGS.mkdir(parents=True, exist_ok=True)
    line = message.rstrip() + "\n"
    with open(LOGS / "driver.log", "a") as handle:
        handle.write(line)
    print(line, end="", flush=True)


def plot_current():
    for start in ("greedy", "random"):
        for spec in study.SPECS:
            runs = []
            for algo in study.ALGOS:
                run = study._load_run(start, spec["name"], algo["key"])
                if run is None:
                    raise SystemExit(f"missing 400-minute run {start} {spec['name']} {algo['key']}")
                run["label"] = algo["label"]
                run["algo_key"] = algo["key"]
                runs.append(run)
            folder = study.OUT / start / spec["name"]
            start_word = "greedy plan" if start == "greedy" else "random plan"
            best_values = [row["best"] for run in runs for row in run["history"]]
            low, high = min(best_values), max(best_values)
            span = max(high - low, 1.0)
            ylim = (low - 0.05 * span, high + 0.05 * span)

            def draw(path, title, limit):
                fig, ax = plt.subplots(figsize=(11, 6.5))
                for run in runs:
                    history = run["history"]
                    ax.plot(
                        study._minutes(history),
                        [row["current"] for row in history],
                        color=study.COLORS[run["algo_key"]],
                        linewidth=1.2,
                        label=run["label"],
                    )
                if limit is not None:
                    ax.set_ylim(*limit)
                study._style(ax, title, "Minutes", "Current objective")
                fig.tight_layout()
                fig.savefig(path, dpi=140)
                plt.close(fig)

            title = f"{study._title(spec)} — current objective, {start_word}"
            draw(folder / "curr_cost_vs_time.png", title, None)
            draw(
                folder / "curr_cost_vs_time_rescaled.png",
                f"{study._title(spec)} — current objective, best-cost axis, {start_word}",
                ylim,
            )
            _log(f"plots {start} {spec['name']}")


def _run_job(start, config_name, algo_key):
    study._apply_weights()
    spec = study.SPEC_BY_NAME[config_name]
    algo = study.ALGO_BY_KEY[algo_key]
    base, _path = load_instance()
    data, meta = build_instance(base, spec)
    trips, greedy_trips = study._trips_for(data, start)
    initial, _aux = evaluate(Solution(data, clone_trips(trips)))
    greedy_cost, _gaux = evaluate(Solution(data, clone_trips(greedy_trips)))
    print_cost(f"{start} {config_name} {algo['label']}", initial)
    run = run_simulated_annealing(
        data,
        iterations=SA_ITERATIONS,
        seed=study.SEED,
        trips=trips,
        t0=algo["t0"],
        t_final=algo["floor"],
        time_limit_sec=EXTEND_SECONDS,
        log_every=50_000,
        alpha=None,
        history_every=study.HISTORY_EVERY,
    )
    run["label"] = algo["label"]
    run["algo_key"] = algo_key
    run["config"] = config_name
    run["start"] = start
    run["greedy_total"] = greedy_cost.total
    summary = {
        "label": algo["label"],
        "algo_key": algo_key,
        "config": config_name,
        "start": start,
        "best": run["best_cost"].total,
        "initial": initial.total,
        "greedy_total": greedy_cost.total,
        "improvement": improvement(initial.total, run["best_cost"].total),
        "improvement_vs_greedy": improvement(greedy_cost.total, run["best_cost"].total),
        "best_iteration": run["best_iteration"],
        "iterations": run["iterations_run"],
        "elapsed": run["elapsed"],
        "stopped": run["stopped"],
        "seconds": EXTEND_SECONDS,
        "t0": algo["t0"],
        "floor": algo["floor"],
        "meta": meta,
    }
    CACHE.mkdir(parents=True, exist_ok=True)
    run.pop("best_trips", None)
    run.pop("accepted_flags", None)
    stem = f"{start}__{config_name}__{algo_key}"
    with open(CACHE / f"{stem}.pkl", "wb") as handle:
        pickle.dump(run, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(CACHE / f"{stem}.json", "w") as handle:
        json.dump(summary, handle)
    _log(
        f"done {start} {config_name} {algo['label']}  "
        f"best={summary['best']:,.2f}  stop={summary['stopped']}  "
        f"iter={summary['best_iteration']}/{summary['iterations']}"
    )


def main():
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    if len(sys.argv) >= 2 and sys.argv[1] == "plots":
        plot_current()
        return
    if len(sys.argv) >= 2 and sys.argv[1] == "job":
        _run_job(sys.argv[2], sys.argv[3], sys.argv[4])
        return
    LOGS.mkdir(parents=True, exist_ok=True)
    (LOGS / "driver.pid").write_text(str(os.getpid()))
    _log(f"Extend {EXTEND_SECONDS}s  jobs={len(JOBS)}  one core each")
    children = []
    for start, config_name, algo_key in JOBS:
        log_path = LOGS / f"{start}__{config_name}__{algo_key}.log"
        handle = open(log_path, "w")
        proc = subprocess.Popen(
            [sys.executable, "-u", str(Path(__file__).resolve()), "job", start, config_name, algo_key],
            stdout=handle,
            stderr=subprocess.STDOUT,
            cwd=str(ROOT),
        )
        children.append((proc, handle, start, config_name, algo_key))
        _log(f"started {start} {config_name} {algo_key} pid={proc.pid}")
    failed = 0
    for proc, handle, start, config_name, algo_key in children:
        code = proc.wait()
        handle.close()
        if code != 0:
            failed += 1
            _log(f"FAILED {start} {config_name} {algo_key} code={code}")
    _log(f"Extend finished  failed={failed}")


if __name__ == "__main__":
    main()
