"""Original instance, greedy start, all 9 algorithms for 100 minutes, 400-minute weights.

    python test/original_100.py           # 9 jobs in parallel, one core each, then plots
    python test/original_100.py plots
    python test/original_100.py job sa_T50
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

import analysis_grid as grid
import analysis_long as study
from calibrate_metaheuristics import improvement, load_instance, print_cost
from heuristic.search_problem import Solution, clone_trips, evaluate
from heuristic.simulated_annealing import run_simulated_annealing
from heuristic.tabu_search import run_tabu_search

CONFIG = "00_original"
START = "greedy"
RUN_SECONDS = 100 * 60
OUT = study.OUT / START / "00_original_100mins"
CACHE = OUT / "_cache"
LOGS = OUT / "logs"
ALGOS = grid.ALGOS
COLORS = grid.COLORS


def _log(message):
    LOGS.mkdir(parents=True, exist_ok=True)
    line = message.rstrip() + "\n"
    with open(LOGS / "driver.log", "a") as handle:
        handle.write(line)
    print(line, end="", flush=True)


def _run_job(algo_key):
    study._apply_weights()
    spec = study.SPEC_BY_NAME[CONFIG]
    algo = grid.ALGO_BY_KEY[algo_key]
    base, _path = load_instance()
    data, meta = grid.build_instance(base, spec)
    trips, greedy_trips = study._trips_for(data, START)
    initial, _aux = evaluate(Solution(data, clone_trips(trips)))
    print_cost(f"{START} {CONFIG} {algo['label']}", initial)
    if algo["family"] == "sa":
        run = run_simulated_annealing(
            data,
            iterations=study.SA_ITERATIONS,
            seed=study.SEED,
            trips=trips,
            t0=algo["t0"],
            t_final=algo["floor"],
            time_limit_sec=RUN_SECONDS,
            log_every=50_000,
            alpha=None,
            history_every=study.HISTORY_EVERY,
        )
    else:
        run = run_tabu_search(
            data,
            iterations=study.TABU_ITERATIONS,
            neighbors=algo["neighbors"],
            tabu_size=algo["tenure"],
            seed=study.SEED,
            trips=trips,
            time_limit_sec=RUN_SECONDS,
            log_every=2_000,
        )
    run["label"] = algo["label"]
    run["algo_key"] = algo_key
    run["config"] = CONFIG
    run["start"] = START
    run["greedy_total"] = initial.total
    summary = study._summarize(run, initial, initial)
    summary.update({
        "label": algo["label"],
        "algo_key": algo_key,
        "family": algo["family"],
        "config": CONFIG,
        "start": START,
        "seconds": RUN_SECONDS,
        "meta": meta,
    })
    for field in ("t0", "floor", "neighbors", "tenure"):
        if field in algo:
            summary[field] = algo[field]
    CACHE.mkdir(parents=True, exist_ok=True)
    run.pop("best_trips", None)
    run.pop("accepted_flags", None)
    with open(CACHE / f"{algo_key}.pkl", "wb") as handle:
        pickle.dump(run, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(CACHE / f"{algo_key}.json", "w") as handle:
        json.dump(summary, handle)
    _log(
        f"done {algo['label']}  best={summary['best']:,.2f}  "
        f"imp={summary['improvement']:.2f}%  iter={summary['best_iteration']}/{summary['iterations']}"
    )


def plot_all():
    runs = []
    for algo in ALGOS:
        path = CACHE / f"{algo['key']}.pkl"
        if not path.exists():
            raise SystemExit(f"missing run {algo['key']}")
        with open(path, "rb") as handle:
            run = pickle.load(handle)
        run["label"] = algo["label"]
        run["algo_key"] = algo["key"]
        runs.append(run)
    conv = OUT / "convergence"
    conv.mkdir(parents=True, exist_ok=True)
    name = study.ENGLISH_TITLES[CONFIG]
    tag = "greedy plan, 100 min"
    initial = runs[0]["initial_cost"].total

    def lines(path, title, ylabel, value, width, limit=None):
        fig, ax = plt.subplots(figsize=(11, 6.5))
        for run in runs:
            history = run["history"]
            ax.plot(
                study._minutes(history),
                [value(row) for row in history],
                color=COLORS[run["algo_key"]],
                linewidth=width,
                label=run["label"],
            )
        if limit is not None:
            ax.set_ylim(*limit)
        study._style(ax, title, "Minutes", ylabel)
        fig.tight_layout()
        fig.savefig(OUT / path, dpi=140)
        plt.close(fig)

    lines("cost_vs_time.png", f"{name} — best objective, {tag}", "Best objective",
          lambda row: row["best"], 2)
    lines("improvement_vs_time.png", f"{name} — improvement vs {tag}", "Improvement (%)",
          lambda row: improvement(initial, row["best"]), 2)
    lines("curr_cost_vs_time.png", f"{name} — current objective, {tag}", "Current objective",
          lambda row: row["current"], 1.2)
    best_values = [row["best"] for run in runs for row in run["history"]]
    low, high = min(best_values), max(best_values)
    span = max(high - low, 1.0)
    lines("curr_cost_vs_time_rescaled.png", f"{name} — current objective, best-cost axis, {tag}",
          "Current objective", lambda row: row["current"], 1.2,
          limit=(low - 0.05 * span, high + 0.05 * span))

    names = ["Start"] + [run["label"] for run in runs]
    totals = [initial] + [run["best_cost"].total for run in runs]
    colors = ["#999999"] + [COLORS[run["algo_key"]] for run in runs]
    fig, ax = plt.subplots(figsize=(11, 6))
    bars = ax.bar(names, totals, color=colors)
    for bar, value in zip(bars, totals):
        ax.annotate(f"{value:,.0f}", xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    ha="center", va="bottom", fontsize=7, rotation=90)
    ax.set_ylim(0, max(totals) * 1.28)
    ax.set_title(f"{name} — final objective, {tag}")
    ax.set_ylabel("Objective value")
    ax.tick_params(axis="x", labelrotation=25)
    ax.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.7)
    fig.tight_layout()
    fig.savefig(OUT / "total_cost.png", dpi=140)
    plt.close(fig)

    for run in runs:
        history = run["history"]
        fig, ax = plt.subplots(figsize=(10, 5.5))
        ax.plot(study._minutes(history), [row["current"] for row in history],
                color="#56B4E9", linewidth=1, label="Current")
        ax.plot(study._minutes(history), [row["best"] for row in history],
                color=COLORS[run["algo_key"]], linewidth=2, label="Best")
        ax.set_title(f"{name} — {run['label']} — {tag}")
        ax.set_xlabel("Minutes")
        ax.set_ylabel("Objective value")
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
        ax.legend()
        fig.tight_layout()
        fig.savefig(conv / f"{run['algo_key']}.png", dpi=140)
        plt.close(fig)
    _log("plots done")


def main():
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    if len(sys.argv) >= 2 and sys.argv[1] == "plots":
        plot_all()
        return
    if len(sys.argv) >= 2 and sys.argv[1] == "job":
        _run_job(sys.argv[2])
        return
    LOGS.mkdir(parents=True, exist_ok=True)
    (LOGS / "driver.pid").write_text(str(os.getpid()))
    _log(f"Original 100 min  jobs={len(ALGOS)}  one core each")
    children = []
    for algo in ALGOS:
        handle = open(LOGS / f"{algo['key']}.log", "w")
        proc = subprocess.Popen(
            [sys.executable, "-u", str(Path(__file__).resolve()), "job", algo["key"]],
            stdout=handle, stderr=subprocess.STDOUT, cwd=str(ROOT),
        )
        children.append((proc, handle, algo["key"]))
        _log(f"started {algo['key']} pid={proc.pid}")
    failed = 0
    for proc, handle, key in children:
        code = proc.wait()
        handle.close()
        if code != 0:
            failed += 1
            _log(f"FAILED {key} code={code}")
    _log(f"finished  failed={failed}")
    if failed == 0:
        plot_all()


if __name__ == "__main__":
    main()
