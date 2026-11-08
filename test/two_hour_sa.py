"""Two-hour greedy SA at the measured T25 and T75.

Cooling matches the 30-minute run: the floor 72.6 is reached at 30 minutes.
The search then stays on that floor until 2 hours. This asks whether the
repair still underway at the half-hour bell only needed more time.

    python test/two_hour_sa.py
    python test/two_hour_sa.py job T25
    python test/two_hour_sa.py plots
"""

import math
import pickle
import subprocess
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

from heuristic.search_problem import MOVE_NAMES, build_initial_trips, set_move_weights
from heuristic.simulated_annealing import run_simulated_annealing
from calibrate_metaheuristics import load_instance, print_cost, save_sa_progress

SEED = 44
RUN_SECONDS = 7200
COOL_SECONDS = 1800
ITERATIONS = 8_000_000
DELTA = 6685.0
T75 = -DELTA / math.log(0.75)
T25 = -DELTA / math.log(0.25)
T_FLOOR = -(DELTA / 10.0) / math.log(0.0001)
JOBS = {"T75": T75, "T25": T25}
COLORS = {"T75": "#0072B2", "T25": "#009E73"}

OUT = ROOT / "plots" / "test"
CACHE = OUT / "_cache"


def _cache_path(label):
    return CACHE / f"greedy_{label}.pkl"


def _run(label):
    data, _path = load_instance()
    set_move_weights([1.0] * len(MOVE_NAMES))
    run = run_simulated_annealing(
        data,
        iterations=ITERATIONS,
        seed=SEED,
        trips=build_initial_trips(data),
        t0=JOBS[label],
        t_final=T_FLOOR,
        time_limit_sec=RUN_SECONDS,
        cool_seconds=COOL_SECONDS,
        log_every=20_000,
        alpha=None,
        history_every=25,
    )
    run["label"] = label
    run.pop("best_trips", None)
    run.pop("accepted_flags", None)
    CACHE.mkdir(parents=True, exist_ok=True)
    with open(_cache_path(label), "wb") as handle:
        pickle.dump(run, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print_cost(f"greedy SA {label}", run["best_cost"])
    _snapshot(label, run)


def _nearest(history, second):
    return min(history, key=lambda row: abs(row["seconds"] - second))


def _snapshot(label, run):
    lines = [f"{label}  stop={run['stopped']}  elapsed={run['elapsed']:.1f}  scored={run['scored']}"]
    for second in (1800, 3600, 5400, 7200):
        row = _nearest(run["history"], second)
        lines.append(
            f"  t={row['seconds']:.0f}  current={row['current']:.2f}  "
            f"best={row['best']:.2f}  T={row['temperature']:.2f}  iter={row['iteration']}"
        )
    lines.append(
        f"  end current={run['history'][-1]['current']:.2f}  "
        f"best={run['best_cost'].total:.2f}  best_iter={run['best_iteration']}  "
        f"T={run['history'][-1]['temperature']:.2f}"
    )
    text = "\n".join(lines)
    print(text, flush=True)
    with open(OUT / "log.txt", "a") as handle:
        handle.write(text + "\n")


def _series(history, key):
    return [row[key] for row in history]


def _mark(ax):
    ax.axvline(COOL_SECONDS, color="#666666", linewidth=1, linestyle=":", label="30 min, floor reached")


def plot_all():
    runs = []
    for label in ("T75", "T25"):
        path = _cache_path(label)
        if not path.exists():
            print(f"missing {path}", flush=True)
            continue
        with open(path, "rb") as handle:
            run = pickle.load(handle)
        run["label"] = label
        runs.append(run)
        folder = OUT / label
        save_sa_progress(run, folder, title=f"Greedy SA {label}, floor at 30 min, search to 2 h")
        # The shared helper has no 30-minute mark. Draw the convergence again into the folder root.
        history = run["history"]
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(_series(history, "seconds"), _series(history, "current"),
                color="#56B4E9", linewidth=1, label="Current")
        ax.plot(_series(history, "seconds"), _series(history, "best"),
                color=COLORS[label], linewidth=2, label="Best")
        twin = ax.twinx()
        twin.plot(_series(history, "seconds"), _series(history, "temperature"),
                  color="#CC79A7", linewidth=1, linestyle="--", label="Temperature")
        twin.set_ylabel("Temperature")
        _mark(ax)
        ax.set_title(f"Greedy SA {label}, floor at 30 min, search to 2 h")
        ax.set_xlabel("Seconds")
        ax.set_ylabel("Objective value")
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = twin.get_legend_handles_labels()
        ax.legend(lines + lines2, labels + labels2, loc="upper right", fontsize=8)
        fig.tight_layout()
        fig.savefig(OUT / f"{label}_convergence.png", dpi=140)
        plt.close(fig)

    if not runs:
        return
    fig, ax = plt.subplots(figsize=(11, 6.5))
    for run in runs:
        history = run["history"]
        color = COLORS[run["label"]]
        ax.plot(_series(history, "seconds"), _series(history, "best"),
                color=color, linewidth=2, label=f"{run['label']} best")
        ax.plot(_series(history, "seconds"), _series(history, "current"),
                color=color, linewidth=1, alpha=0.85, label=f"{run['label']} current")
    _mark(ax)
    ax.set_title("Greedy SA T25 and T75 — best and current, 2 hours")
    ax.set_xlabel("Seconds")
    ax.set_ylabel("Objective value")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(OUT / "cost_vs_time.png", dpi=140)
    plt.close(fig)
    print(f"plots written under {OUT}", flush=True)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if len(sys.argv) >= 3 and sys.argv[1] == "job":
        _run(sys.argv[2])
        return
    if len(sys.argv) >= 2 and sys.argv[1] == "plots":
        plot_all()
        return
    with open(OUT / "log.txt", "w") as handle:
        handle.write(
            f"T75={T75:.2f}  T25={T25:.2f}  floor={T_FLOOR:.2f}  "
            f"cool={COOL_SECONDS}s  run={RUN_SECONDS}s  seed={SEED}  greedy\n"
        )
    procs = []
    handles = []
    for label in ("T75", "T25"):
        handle = open(OUT / f"{label}.log", "w")
        handles.append(handle)
        procs.append(subprocess.Popen(
            [sys.executable, "-u", str(Path(__file__).resolve()), "job", label],
            cwd=str(ROOT),
            stdout=handle,
            stderr=subprocess.STDOUT,
        ))
        print(f"started {label} pid={procs[-1].pid}", flush=True)
    codes = [proc.wait() for proc in procs]
    for handle in handles:
        handle.close()
    print(f"exit codes {codes}", flush=True)
    plot_all()
    if any(code != 0 for code in codes):
        sys.exit(1)


if __name__ == "__main__":
    main()
