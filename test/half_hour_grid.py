"""Half-hour SA and tabu grids on the greedy and random starts.

SA temperatures are the measured-move temperatures, plus 500 and 50.
T75, T50, T25 and 500 cool on the clock to the shared floor 72.6.
T0=50 is already below that floor, so it cools by the same factor as T25.

Tabu is neighbors 35, 70, 150 crossed with tenure 25, 125, 625.
Comparison plots for all nine tabu settings sit in the greedy and random
folders. The best SA and the best tabu of each start are plotted against
each other from these runs, without a second execution.

    python test/half_hour_grid.py
"""

import math
import pickle
import random
import subprocess
import sys
import time
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
    MOVE_NAMES,
)
from heuristic.simulated_annealing import run_simulated_annealing
from heuristic.tabu_search import run_tabu_search
from calibrate_metaheuristics import improvement, print_cost, save_sa_progress
from tabu_vs_sa import save_comparison

SEED = 44
HALF = 1800
SA_ITERATIONS = 5_000_000
TABU_ITERATIONS = 1_000_000
HISTORY_EVERY = 25
WORKERS = 8
DELTA = 6685.0
T75 = -DELTA / math.log(0.75)
T50 = -DELTA / math.log(0.50)
T25 = -DELTA / math.log(0.25)
T_FLOOR = -(DELTA / 10.0) / math.log(0.0001)
RATIO = T_FLOOR / T25

SA_JOBS = (
    ("T75", T75, T_FLOOR),
    ("T50", T50, T_FLOOR),
    ("T25", T25, T_FLOOR),
    ("500", 500.0, T_FLOOR),
    ("50", 50.0, 50.0 * RATIO),
)
TABU_NEIGHBORS = (35, 70, 150)
TABU_TENURES = (25, 125, 625)

SA_COLORS = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7"]
TABU_COLORS = [
    "#0072B2", "#56B4E9", "#4C78A8",
    "#E69F00", "#F0E442", "#D55E00",
    "#009E73", "#CC79A7", "#882255",
]

SA_OUT = ROOT / "plots" / "sa"
TABU_OUT = ROOT / "plots" / "tabu"
VS_OUT = ROOT / "plots" / "tabu_vs_sa"
LOG = ROOT / "plots" / "half_hour_log.txt"


def _log(text):
    print(text, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG, "a") as handle:
        handle.write(text + "\n")
        handle.flush()


def _cache(kind, key):
    folder = SA_OUT if kind == "sa" else TABU_OUT
    return folder / "_cache" / f"{key}.pkl"


def _load(kind, key):
    path = _cache(kind, key)
    if not path.exists():
        return None
    with open(path, "rb") as handle:
        return pickle.load(handle)


def _store(kind, key, run):
    path = _cache(kind, key)
    path.parent.mkdir(parents=True, exist_ok=True)
    run.pop("best_trips", None)
    run.pop("accepted_flags", None)
    with open(path, "wb") as handle:
        pickle.dump(run, handle, protocol=pickle.HIGHEST_PROTOCOL)


def _trips(data, start):
    if start == "greedy":
        return build_initial_trips(data)
    return build_random_trips(data, random.Random(SEED))


def _sa_key(start, label):
    return f"{start}_{label}"


def _tabu_key(start, neighbors, tenure):
    return f"{start}_n{neighbors}_t{tenure}"


def _pending():
    jobs = []
    for start in ("greedy", "random"):
        for label, _t0, _floor in SA_JOBS:
            if _load("sa", _sa_key(start, label)) is None:
                jobs.append(("sa", start, label))
        for neighbors in TABU_NEIGHBORS:
            for tenure in TABU_TENURES:
                if _load("tabu", _tabu_key(start, neighbors, tenure)) is None:
                    jobs.append(("tabu", start, str(neighbors), str(tenure)))
    return jobs


def _run_sa(start, label):
    from calibrate_metaheuristics import load_instance
    data, _path = load_instance()
    set_move_weights([1.0] * len(MOVE_NAMES))
    spec = {name: (t0, floor) for name, t0, floor in SA_JOBS}[label]
    t0, floor = spec
    trips = _trips(data, start)
    run = run_simulated_annealing(
        data,
        iterations=SA_ITERATIONS,
        seed=SEED,
        trips=trips,
        t0=t0,
        t_final=floor,
        time_limit_sec=HALF,
        log_every=20_000,
        alpha=None,
        history_every=HISTORY_EVERY,
    )
    run["label"] = label
    _store("sa", _sa_key(start, label), run)


def _run_tabu(start, neighbors, tenure):
    from calibrate_metaheuristics import load_instance
    data, _path = load_instance()
    set_move_weights([1.0] * len(MOVE_NAMES))
    trips = _trips(data, start)
    run = run_tabu_search(
        data,
        iterations=TABU_ITERATIONS,
        neighbors=neighbors,
        tabu_size=tenure,
        seed=SEED,
        trips=trips,
        time_limit_sec=HALF,
        log_every=1_000,
    )
    run["label"] = f"n={neighbors}, t={tenure}"
    _store("tabu", _tabu_key(start, neighbors, tenure), run)


def _job_main(argv):
    kind = argv[0]
    start = argv[1]
    if kind == "sa":
        _run_sa(start, argv[2])
    else:
        _run_tabu(start, int(argv[2]), int(argv[3]))


def _series(history, key):
    return [row[key] for row in history]


def _style(ax, title, xlabel, ylabel):
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    ax.legend(fontsize=8, ncol=2)


def save_group(runs, out_dir, colors, title, baseline_name):
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 6.5))
    for run, color in zip(runs, colors):
        ax.plot(_series(run["history"], "seconds"), _series(run["history"], "best"),
                color=color, linewidth=2, label=run["label"])
    _style(ax, f"{title} — best objective vs time", "Seconds", "Best objective")
    fig.tight_layout()
    fig.savefig(out_dir / "cost_vs_time.png", dpi=140)
    plt.close(fig)

    initial = runs[0]["initial_cost"].total
    fig, ax = plt.subplots(figsize=(11, 6.5))
    for run, color in zip(runs, colors):
        ax.plot(
            _series(run["history"], "seconds"),
            [improvement(initial, value) for value in _series(run["history"], "best")],
            color=color, linewidth=2, label=run["label"],
        )
    _style(ax, f"{title} — improvement vs {baseline_name.lower()}", "Seconds", "Improvement (%)")
    fig.tight_layout()
    fig.savefig(out_dir / "improvement_vs_time.png", dpi=140)
    plt.close(fig)

    names = [baseline_name] + [run["label"] for run in runs]
    costs = [runs[0]["initial_cost"]] + [run["best_cost"] for run in runs]
    bar_colors = ["#999999"] + list(colors[:len(runs)])
    fig, ax = plt.subplots(figsize=(12, 6))
    totals = [cost.total for cost in costs]
    bars = ax.bar(names, totals, color=bar_colors)
    for bar, value in zip(bars, totals):
        ax.annotate(f"{value:,.0f}", xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    ha="center", va="bottom", fontsize=7, rotation=90)
    ax.set_title(f"{title} — final objective")
    ax.set_ylabel("Objective value")
    ax.tick_params(axis="x", labelrotation=30)
    ax.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.7)
    fig.tight_layout()
    fig.savefig(out_dir / "total_cost.png", dpi=140)
    plt.close(fig)

    metrics = [
        ("transport", "Transport"),
        ("holding", "Holding"),
        ("backorder", "Backorder"),
        ("unsold", "End-of-month unsold"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    for ax, (attr, panel) in zip(axes.ravel(), metrics):
        values = [getattr(cost, attr) for cost in costs]
        ax.bar(names, values, color=bar_colors)
        ax.set_title(panel)
        ax.tick_params(axis="x", labelrotation=30)
        ax.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.7)
    fig.suptitle(f"{title} — cost components")
    fig.tight_layout()
    fig.savefig(out_dir / "components.png", dpi=140)
    plt.close(fig)


def save_tabu_run(run, out_dir, title):
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


def save_diagnostics(runs, out_dir):
    labels = [run["label"] for run in runs]
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    axes[0].bar(labels, [run["forbidden_share"] for run in runs], color=TABU_COLORS)
    axes[0].set_title("Forbidden share (%)")
    axes[1].bar(labels, [run["worse_steps"] for run in runs], color=TABU_COLORS)
    axes[1].set_title("Worse steps")
    axes[2].bar(labels, [run["all_tabu_iterations"] for run in runs], color=TABU_COLORS)
    axes[2].set_title("All-tabu iterations")
    for ax in axes:
        ax.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.7)
        ax.tick_params(axis="x", labelrotation=40)
    fig.suptitle("Tabu memory diagnostics")
    fig.tight_layout()
    fig.savefig(out_dir / "diagnostics.png", dpi=140)
    plt.close(fig)


def _plot():
    for start, baseline in (("greedy", "Greedy"), ("random", "Random")):
        sa_runs = []
        for label, t0, floor in SA_JOBS:
            run = _load("sa", _sa_key(start, label))
            if run is None:
                raise SystemExit(f"missing SA cache {start} {label}")
            run["label"] = label
            sa_runs.append(run)
            print_cost(f"{start} SA {label}", run["best_cost"])
            _log(
                f"{start} SA {label}  T0={t0:,.2f}  floor={floor:,.2f}  "
                f"best={run['best_cost'].total:,.2f}  "
                f"improvement={improvement(run['initial_cost'].total, run['best_cost'].total):.2f}%  "
                f"iter={run['best_iteration']}/{run['iterations_run']}"
            )
            save_sa_progress(run, SA_OUT / start / label, f"SA {label}")
        save_group(sa_runs, SA_OUT / start, SA_COLORS, f"SA {start}", baseline)

        tabu_runs = []
        for neighbors in TABU_NEIGHBORS:
            for tenure in TABU_TENURES:
                run = _load("tabu", _tabu_key(start, neighbors, tenure))
                if run is None:
                    raise SystemExit(f"missing tabu cache {start} n={neighbors} t={tenure}")
                run["label"] = f"n={neighbors}, t={tenure}"
                tabu_runs.append(run)
                _log(
                    f"{start} tabu n={neighbors} t={tenure}  "
                    f"best={run['best_cost'].total:,.2f}  "
                    f"improvement={improvement(run['initial_cost'].total, run['best_cost'].total):.2f}%  "
                    f"forbidden={run['forbidden_share']:.3f}%"
                )
                save_tabu_run(
                    run,
                    TABU_OUT / start / f"n{neighbors}_t{tenure}",
                    f"Tabu n={neighbors}, tenure={tenure}",
                )
        save_group(tabu_runs, TABU_OUT / start, TABU_COLORS, f"Tabu {start}", baseline)
        save_diagnostics(tabu_runs, TABU_OUT / start)

        best_sa = min(sa_runs, key=lambda item: item["best_cost"].total)
        best_tabu = min(tabu_runs, key=lambda item: item["best_cost"].total)
        out = VS_OUT / start
        save_comparison(best_sa, best_tabu, out, baseline.lower())
        note = (
            f"{start}\n"
            f"SA {best_sa['label']}  {best_sa['best_cost'].total:,.2f}\n"
            f"Tabu {best_tabu['label']}  {best_tabu['best_cost'].total:,.2f}\n"
        )
        out.mkdir(parents=True, exist_ok=True)
        (out / "chosen.txt").write_text(note)
        _log(f"1v1 {start}: SA {best_sa['label']} vs Tabu {best_tabu['label']}")
        _log(f"Plots: {out}")


def _launch():
    jobs = _pending()
    _log(f"Pending jobs: {len(jobs)}")
    _log(
        f"T75={T75:,.2f}  T50={T50:,.2f}  T25={T25:,.2f}  "
        f"floor={T_FLOOR:,.2f}  T0=50 floor={50.0 * RATIO:,.2f}  "
        f"each {HALF:.0f}s  workers={WORKERS}"
    )
    if not jobs:
        _plot()
        _log("Nothing left to run. Plots refreshed.")
        return
    running = []
    while jobs or running:
        while jobs and len(running) < WORKERS:
            job = jobs.pop(0)
            log_path = LOG.parent / "_job_logs" / ("_".join(job) + ".log")
            log_path.parent.mkdir(parents=True, exist_ok=True)
            handle = open(log_path, "w")
            proc = subprocess.Popen(
                [sys.executable, str(Path(__file__).resolve()), "job", *job],
                cwd=str(ROOT),
                stdout=handle,
                stderr=subprocess.STDOUT,
            )
            running.append((proc, handle, job, time.time()))
            _log(f"start {' '.join(job)}  pid={proc.pid}")
        time.sleep(10)
        still = []
        for proc, handle, job, started in running:
            code = proc.poll()
            if code is None:
                still.append((proc, handle, job, started))
                continue
            handle.close()
            elapsed = time.time() - started
            if code == 0:
                _log(f"done {' '.join(job)}  {elapsed / 60:.1f} min")
            else:
                _log(f"FAILED {' '.join(job)}  code={code}  {elapsed / 60:.1f} min")
        running = still
    failed = []
    for job in _all_jobs():
        kind = job[0]
        key = _sa_key(job[1], job[2]) if kind == "sa" else _tabu_key(job[1], int(job[2]), int(job[3]))
        if _load(kind, key) is None:
            failed.append(job)
    if failed:
        _log(f"Missing after the pool: {failed}")
        raise SystemExit(1)
    _plot()
    _log("Finished.")


def _all_jobs():
    jobs = []
    for start in ("greedy", "random"):
        for label, _t0, _floor in SA_JOBS:
            jobs.append(("sa", start, label))
        for neighbors in TABU_NEIGHBORS:
            for tenure in TABU_TENURES:
                jobs.append(("tabu", start, str(neighbors), str(tenure)))
    return jobs


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "job":
        _job_main(sys.argv[2:])
    elif len(sys.argv) > 1 and sys.argv[1] == "plots":
        _plot()
    else:
        _launch()
