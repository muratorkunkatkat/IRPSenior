"""1000-minute extensions. Does not touch the 400-minute cache.

    python test/extend_1000.py plots
    python test/extend_1000.py extplots
    python test/extend_1000.py combined
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


def _load_extended(start, config_name, algo_key):
    path = CACHE / f"{start}__{config_name}__{algo_key}.pkl"
    if not path.exists():
        raise SystemExit(f"missing 1000-minute run {start} {config_name} {algo_key}")
    with open(path, "rb") as handle:
        run = pickle.load(handle)
    run["label"] = study.ALGO_BY_KEY[algo_key]["label"]
    run["algo_key"] = algo_key
    return run


def plot_extended():
    groups = {}
    for start, config_name, algo_key in JOBS:
        groups.setdefault((start, config_name), []).append(algo_key)
    order = [algo["key"] for algo in study.ALGOS]
    for (start, config_name), keys in groups.items():
        keys.sort(key=order.index)
        runs = [_load_extended(start, config_name, key) for key in keys]
        spec = study.SPEC_BY_NAME[config_name]
        name = study._title(spec)
        folder = OUT / start / config_name
        conv = folder / "convergence"
        conv.mkdir(parents=True, exist_ok=True)
        initial = runs[0]["initial_cost"].total
        greedy_total = runs[0].get("greedy_total", initial)
        start_word = "greedy plan" if start == "greedy" else "random plan"
        tag = "1000 min"

        def lines(path, title, ylabel, value, width, limit=None, greedy_line=False):
            fig, ax = plt.subplots(figsize=(11, 6.5))
            for run in runs:
                history = run["history"]
                ax.plot(
                    study._minutes(history),
                    [value(row) for row in history],
                    color=study.COLORS[run["algo_key"]],
                    linewidth=width,
                    label=run["label"],
                )
            if greedy_line:
                ax.axhline(greedy_total, color="#666666", linestyle=":", linewidth=1.2, label="Greedy plan")
            if limit is not None:
                ax.set_ylim(*limit)
            study._style(ax, title, "Minutes", ylabel)
            fig.tight_layout()
            fig.savefig(folder / path, dpi=140)
            plt.close(fig)

        lines(
            "cost_vs_time.png",
            f"{name} — best objective, {start_word}, {tag}",
            "Best objective",
            lambda row: row["best"],
            2,
            greedy_line=(start == "random"),
        )
        lines(
            "improvement_vs_time.png",
            f"{name} — improvement vs {start_word}, {tag}",
            "Improvement (%)",
            lambda row: improvement(initial, row["best"]),
            2,
        )
        lines(
            "curr_cost_vs_time.png",
            f"{name} — current objective, {start_word}, {tag}",
            "Current objective",
            lambda row: row["current"],
            1.2,
        )
        best_values = [row["best"] for run in runs for row in run["history"]]
        low, high = min(best_values), max(best_values)
        span = max(high - low, 1.0)
        lines(
            "curr_cost_vs_time_rescaled.png",
            f"{name} — current objective, best-cost axis, {start_word}, {tag}",
            "Current objective",
            lambda row: row["current"],
            1.2,
            limit=(low - 0.05 * span, high + 0.05 * span),
        )

        names = ["Start"] + [run["label"] for run in runs]
        totals = [initial] + [run["best_cost"].total for run in runs]
        colors = ["#999999"] + [study.COLORS[run["algo_key"]] for run in runs]
        fig, ax = plt.subplots(figsize=(11, 6))
        bars = ax.bar(names, totals, color=colors)
        for bar, value in zip(bars, totals):
            ax.annotate(
                f"{value:,.0f}",
                xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                ha="center",
                va="bottom",
                fontsize=7,
                rotation=90,
            )
        ax.set_ylim(0, max(totals) * 1.28)
        ax.set_title(f"{name} — final objective, {start_word}, {tag}")
        ax.set_ylabel("Objective value")
        ax.tick_params(axis="x", labelrotation=25)
        ax.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.7)
        fig.tight_layout()
        fig.savefig(folder / "total_cost.png", dpi=140)
        plt.close(fig)

        for run in runs:
            history = run["history"]
            fig, ax = plt.subplots(figsize=(10, 5.5))
            ax.plot(study._minutes(history), [row["current"] for row in history],
                    color="#56B4E9", linewidth=1, label="Current")
            ax.plot(study._minutes(history), [row["best"] for row in history],
                    color=study.COLORS[run["algo_key"]], linewidth=2, label="Best")
            ax.set_title(f"{name} — {run['label']} — {start_word}, {tag}")
            ax.set_xlabel("Minutes")
            ax.set_ylabel("Objective value")
            ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
            ax.legend()
            fig.tight_layout()
            fig.savefig(conv / f"{run['algo_key']}.png", dpi=140)
            plt.close(fig)
        _log(f"extended plots {start} {config_name}")


def plot_combined():
    """Best objective of all six algorithms; extended ones drawn with their 1000-minute run."""
    groups = {}
    for start, config_name, algo_key in JOBS:
        groups.setdefault((start, config_name), set()).add(algo_key)
    for (start, config_name), extended in groups.items():
        spec = study.SPEC_BY_NAME[config_name]
        start_word = "greedy plan" if start == "greedy" else "random plan"
        fig, ax = plt.subplots(figsize=(11, 6.5))
        greedy_total = None
        for algo in study.ALGOS:
            key = algo["key"]
            if key in extended:
                run = _load_extended(start, config_name, key)
                label = f"{algo['label']} (1000 min)"
            else:
                run = study._load_run(start, config_name, key)
                if run is None:
                    raise SystemExit(f"missing 400-minute run {start} {config_name} {key}")
                label = f"{algo['label']} (400 min)"
            greedy_total = run.get("greedy_total", greedy_total)
            history = run["history"]
            ax.plot(
                study._minutes(history),
                [row["best"] for row in history],
                color=study.COLORS[key],
                linewidth=2,
                label=label,
            )
        ax.axvline(400, color="#999999", linestyle="--", linewidth=1)
        if start == "random" and greedy_total is not None:
            ax.axhline(greedy_total, color="#666666", linestyle=":", linewidth=1.2, label="Greedy plan")
        study._style(
            ax,
            f"{study._title(spec)} — best objective, {start_word}, 400 + 1000 min",
            "Minutes",
            "Best objective",
        )
        fig.tight_layout()
        folder = OUT / start / config_name
        folder.mkdir(parents=True, exist_ok=True)
        fig.savefig(folder / "cost_vs_time_combined.png", dpi=140)
        plt.close(fig)
        _log(f"combined plot {start} {config_name}")


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
    if len(sys.argv) >= 2 and sys.argv[1] == "extplots":
        plot_extended()
        plot_combined()
        return
    if len(sys.argv) >= 2 and sys.argv[1] == "combined":
        plot_combined()
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
