"""400-minute study after the ALNS weight pilot.

One 400-minute equal-weight SA at T25, random start, sets the weights.
Then four instances, six algorithms, greedy start and random start, 400 minutes each.

    python test/analysis_long.py
    python test/analysis_long.py job greedy 00_original sa_T25
    python test/analysis_long.py plots
    python test/analysis_long.py doc
"""

import json
import math
import os
import pickle
import signal
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

from analysis_grid import CONFIGS, build_instance
from calibrate_metaheuristics import improvement, load_instance, print_cost
from heuristic.search_problem import (
    MOVE_NAMES,
    Solution,
    bounded_score_weights,
    build_initial_trips,
    build_random_trips,
    clone_trips,
    evaluate,
    move_stats,
    reset_move_stats,
    set_move_weights,
)
from heuristic.simulated_annealing import run_simulated_annealing
from heuristic.tabu_search import run_tabu_search

SEED = 44
CALIBRATE_SECONDS = 24000
RUN_SECONDS = 24000
WORKERS = 8
SA_ITERATIONS = 20_000_000
TABU_ITERATIONS = 2_000_000
HISTORY_EVERY = 100
MARKS = (6000, 12000, 18000, 24000)

DELTA = 6685.0
T75 = -DELTA / math.log(0.75)
T25 = -DELTA / math.log(0.25)
T_FLOOR = -(DELTA / 10.0) / math.log(0.0001)
RATIO = T_FLOOR / T25

CHOSEN = (
    "00_original",
    "02_growth_no_fleet",
    "07_full_slack_thin",
    "09_far_tight",
)
SA_JOBS = (
    ("T75", T75, T_FLOOR),
    ("T25", T25, T_FLOOR),
    ("500", 500.0, T_FLOOR),
    ("50", 50.0, 50.0 * RATIO),
)
TABU_JOBS = (
    (70, 25),
    (70, 625),
)

OUT = ROOT / "plots" / "analysis"
LOGS = OUT / "logs"
CACHE = OUT / "_cache"
WEIGHTS_PATH = OUT / "weights.json"
DRIVER_LOG = LOGS / "driver.log"

COLORS = {
    "sa_T75": "#0072B2",
    "sa_T25": "#009E73",
    "sa_500": "#D55E00",
    "sa_50": "#CC79A7",
    "tabu_n70_t25": "#332288",
    "tabu_n70_t625": "#000000",
}


def _algos():
    jobs = []
    for label, t0, floor in SA_JOBS:
        jobs.append({
            "key": f"sa_{label}",
            "label": f"SA {label}",
            "family": "sa",
            "t0": t0,
            "floor": floor,
        })
    for neighbors, tenure in TABU_JOBS:
        jobs.append({
            "key": f"tabu_n{neighbors}_t{tenure}",
            "label": f"Tabu {neighbors}/{tenure}",
            "family": "tabu",
            "neighbors": neighbors,
            "tenure": tenure,
        })
    return jobs


ALGOS = _algos()
ALGO_BY_KEY = {item["key"]: item for item in ALGOS}
SPECS = [spec for spec in CONFIGS if spec["name"] in CHOSEN]
SPEC_BY_NAME = {spec["name"]: spec for spec in SPECS}
ENGLISH_TITLES = {
    "00_original": "Original",
    "02_growth_no_fleet": "Growth, same fleet",
    "07_full_slack_thin": "Full map, slack",
    "09_far_tight": "Far, tight",
}
INITIAL = ROOT / "plots" / "initial"


def _title(spec):
    return ENGLISH_TITLES.get(spec["name"], spec["name"])


def _log(text):
    print(text, flush=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    with open(DRIVER_LOG, "a") as handle:
        handle.write(text + "\n")
        handle.flush()


def _cost_dict(cost):
    return {
        "total": cost.total,
        "transport": cost.transport,
        "holding": cost.holding,
        "backorder": cost.backorder,
        "unsold": cost.unsold,
        "capacity": cost.capacity,
        "delivered": cost.delivered,
        "holding_units": cost.holding_units,
        "backorder_units": cost.backorder_units,
        "unsold_units": cost.unsold_units,
    }


def _snapshot(history, second):
    return min(history, key=lambda row: abs(row["seconds"] - second))


def _summary_path(start, config_name, algo_key):
    return CACHE / f"{start}__{config_name}__{algo_key}.json"


def _cache_path(start, config_name, algo_key):
    return CACHE / f"{start}__{config_name}__{algo_key}.pkl"


def _load_summary(start, config_name, algo_key):
    path = _summary_path(start, config_name, algo_key)
    if not path.exists():
        return None
    with open(path) as handle:
        return json.load(handle)


def _summarize(run, initial, greedy_cost):
    history = run["history"]
    best = run["best_cost"]
    row = {
        "best": best.total,
        "initial": initial.total,
        "greedy_total": greedy_cost.total,
        "improvement": improvement(initial.total, best.total),
        "improvement_vs_greedy": improvement(greedy_cost.total, best.total),
        "best_iteration": run["best_iteration"],
        "iterations": run["iterations_run"],
        "elapsed": run["elapsed"],
        "stopped": run["stopped"],
        "end_current": history[-1]["current"],
        "end_temperature": history[-1].get("temperature"),
        "scored": run.get("scored", run.get("scored_candidates")),
        "move_counts": run.get("move_counts", {}),
        "forbidden_share": run.get("forbidden_share"),
        "worse_steps": run.get("worse_steps"),
        "all_tabu_iterations": run.get("all_tabu_iterations"),
        **_cost_dict(best),
    }
    for second in MARKS:
        if history[-1]["seconds"] + 1 < second and run["elapsed"] + 1 < second:
            continue
        mark = _snapshot(history, second)
        row[f"best_at_{second}"] = mark["best"]
        row[f"current_at_{second}"] = mark["current"]
    return row


def _apply_weights():
    with open(WEIGHTS_PATH) as handle:
        payload = json.load(handle)
    set_move_weights(payload["weights"])
    return payload


def _trips_for(data, start):
    greedy = build_initial_trips(data)
    if start == "greedy":
        return greedy, greedy
    return build_random_trips(data, random.Random(SEED)), greedy


def _run_job(start, config_name, algo_key):
    _apply_weights()
    spec = SPEC_BY_NAME[config_name]
    algo = ALGO_BY_KEY[algo_key]
    base, _path = load_instance()
    data, meta = build_instance(base, spec)
    trips, greedy_trips = _trips_for(data, start)
    initial, _aux = evaluate(Solution(data, clone_trips(trips)))
    greedy_cost, _gaux = evaluate(Solution(data, clone_trips(greedy_trips)))
    print_cost(f"{start} {config_name} start", initial)
    if algo["family"] == "sa":
        run = run_simulated_annealing(
            data,
            iterations=SA_ITERATIONS,
            seed=SEED,
            trips=trips,
            t0=algo["t0"],
            t_final=algo["floor"],
            time_limit_sec=RUN_SECONDS,
            log_every=50_000,
            alpha=None,
            history_every=HISTORY_EVERY,
        )
    else:
        run = run_tabu_search(
            data,
            iterations=TABU_ITERATIONS,
            neighbors=algo["neighbors"],
            tabu_size=algo["tenure"],
            seed=SEED,
            trips=trips,
            time_limit_sec=RUN_SECONDS,
            log_every=2_000,
        )
    run["label"] = algo["label"]
    run["algo_key"] = algo_key
    run["config"] = config_name
    run["start"] = start
    run["greedy_total"] = greedy_cost.total
    summary = _summarize(run, initial, greedy_cost)
    summary.update({
        "label": algo["label"],
        "algo_key": algo_key,
        "family": algo["family"],
        "config": config_name,
        "start": start,
        "greedy": _cost_dict(greedy_cost),
        "start_cost": _cost_dict(initial),
        "initial_tours": len(trips),
        "meta": meta,
    })
    if algo["family"] == "sa":
        summary["t0"] = algo["t0"]
        summary["floor"] = algo["floor"]
    else:
        summary["neighbors"] = algo["neighbors"]
        summary["tenure"] = algo["tenure"]
    CACHE.mkdir(parents=True, exist_ok=True)
    run.pop("best_trips", None)
    run.pop("accepted_flags", None)
    with open(_cache_path(start, config_name, algo_key), "wb") as handle:
        pickle.dump(run, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(_summary_path(start, config_name, algo_key), "w") as handle:
        json.dump(summary, handle)
    _log(
        f"done {start} {config_name} {algo['label']}  "
        f"best={summary['best']:,.2f}  imp={summary['improvement']:.2f}%  "
        f"iter={summary['best_iteration']}/{summary['iterations']}"
    )


def _weights_ready():
    if not WEIGHTS_PATH.exists():
        return False
    payload = json.loads(WEIGHTS_PATH.read_text())
    return (
        payload.get("method") == "alns-bounded"
        and payload.get("seconds") == CALIBRATE_SECONDS
        and len(payload.get("weights", [])) == len(MOVE_NAMES)
    )


def _minutes(history):
    return [row["seconds"] / 60.0 for row in history]


def _save_initial_plots(run, stats, payload):
    from calibrate_metaheuristics import save_move_acceptance

    INITIAL.mkdir(parents=True, exist_ok=True)
    for stale in ("weights.json", "weights.txt", "reweight.log"):
        path = INITIAL / stale
        if path.exists():
            path.unlink()
    history = run["history"]
    minutes = _minutes(history)
    title = "Weight pilot, random start, equal weights, T25, 400 minutes"

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(minutes, [row["current"] for row in history], color="#56B4E9", linewidth=1, label="Current")
    ax.plot(minutes, [row["best"] for row in history], color="#0072B2", linewidth=2, label="Best")
    twin = ax.twinx()
    twin.plot(minutes, [row["temperature"] for row in history], color="#CC79A7", linewidth=1, linestyle="--", label="Temperature")
    twin.set_ylabel("Temperature")
    ax.set_title(title)
    ax.set_xlabel("Minutes")
    ax.set_ylabel("Objective value")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = twin.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc="upper right")
    fig.tight_layout()
    fig.savefig(INITIAL / "sa_convergence.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(minutes, [row["best_transport"] for row in history], color="#0072B2", linewidth=2, label="Transport")
    ax.plot(minutes, [row["best_backorder"] for row in history], color="#D55E00", linewidth=2, label="Backorder")
    ax.plot(minutes, [row["best_holding"] for row in history], color="#009E73", linewidth=2, label="Holding")
    ax.plot(minutes, [row["best_unsold"] for row in history], color="#CC79A7", linewidth=2, label="Unsold")
    _style(ax, "Weight pilot — pieces of the best cost", "Minutes", "Cost")
    fig.tight_layout()
    fig.savefig(INITIAL / "sa_components_vs_time.png", dpi=140)
    plt.close(fig)

    save_move_acceptance(run, stats, INITIAL)

    names = payload["names"]
    weights = payload["weights"]
    total = sum(weights) or 1.0
    fig, ax = plt.subplots(figsize=(10, 5.8))
    bars = ax.bar(names, weights, color="#0072B2")
    for bar, weight in zip(bars, weights):
        ax.annotate(
            f"{weight:.2f}\n{100.0 * weight / total:.0f}%",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            ha="center", va="bottom", fontsize=8,
        )
    ax.set_ylim(0, 5.8)
    ax.set_title("Move weights after the 400-minute pilot")
    ax.set_ylabel("Weight (1 to 5)")
    ax.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.7)
    fig.tight_layout()
    fig.savefig(INITIAL / "weights.png", dpi=140)
    plt.close(fig)
    _log(f"Initial plots: {INITIAL}")


def run_calibration():
    if _weights_ready():
        _log(f"Weights already at {WEIGHTS_PATH}")
        return
    data, path = load_instance()
    set_move_weights([1.0] * len(MOVE_NAMES))
    reset_move_stats()
    trips = build_random_trips(data, random.Random(SEED))
    initial, _aux = evaluate(Solution(data, clone_trips(trips)))
    _log(
        f"Calibration {path}  random start  seed {SEED}  equal weights  "
        f"T0={T25:,.2f}  floor={T_FLOOR:,.2f}  {CALIBRATE_SECONDS}s"
    )
    print_cost("Random start", initial)
    run = run_simulated_annealing(
        data,
        iterations=SA_ITERATIONS,
        seed=SEED,
        trips=trips,
        t0=T25,
        t_final=T_FLOOR,
        time_limit_sec=CALIBRATE_SECONDS,
        log_every=50_000,
        alpha=None,
        history_every=HISTORY_EVERY,
    )
    print_cost("Calibration best", run["best_cost"])
    outcomes = run["outcome_counts"]
    weights = bounded_score_weights(outcomes)
    stats = move_stats()
    payload = {
        "method": "alns-bounded",
        "source": "Ropke and Pisinger, Transportation Science 2006, scores 33 / 9 / 13",
        "scores": {"new_best": 33, "improved": 9, "accepted_worse": 13, "rejected": 0},
        "bound": [1.0, 5.0],
        "weights": weights,
        "names": list(MOVE_NAMES),
        "outcomes": outcomes,
        "stats": stats,
        "move_counts": run.get("move_counts", {}),
        "proposed_counts": run.get("proposed_counts", {}),
        "t0": T25,
        "floor": T_FLOOR,
        "seed": SEED,
        "seconds": CALIBRATE_SECONDS,
        "start": "random",
        "initial_total": initial.total,
        "best_total": run["best_cost"].total,
        "improvement": improvement(initial.total, run["best_cost"].total),
        "iterations": run["iterations_run"],
        "best_iteration": run["best_iteration"],
        "elapsed": run["elapsed"],
    }
    OUT.mkdir(parents=True, exist_ok=True)
    tmp = WEIGHTS_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(WEIGHTS_PATH)
    _write_weight_note(payload)
    _save_initial_plots(run, stats, payload)
    _log("Weights: " + ", ".join(f"{name}={weight:.3f}" for name, weight in zip(MOVE_NAMES, weights)))


def _write_weight_note(payload):
    total = sum(payload["weights"]) or 1.0
    lines = [
        "Pilot: original instance, random start, seed 44, equal weights, T25, 400 minutes.",
        "Score per scored neighbor: new best 33, better than current 9, accepted worse 13, rejected 0.",
        "Raw weight = total score / scored neighbors. Divide by the smallest positive raw value, then cap at 5.",
        "A move with no score stays at 1. Build rate is recorded and not used.",
        "Temperatures are not taken from this pilot.",
        "",
    ]
    for name, weight in zip(payload["names"], payload["weights"]):
        outcome = payload["outcomes"][name]
        uses = sum(outcome.values())
        lines.append(
            f"{name}\t{weight:.6f}\tdraw={100.0 * weight / total:.2f}%\t"
            f"uses={uses}\tnew_best={outcome['new_best']}\timproved={outcome['improved']}\t"
            f"accepted_worse={outcome['accepted_worse']}\trejected={outcome['rejected']}"
        )
    (OUT / "weights.txt").write_text("\n".join(lines) + "\n")


def _pending():
    jobs = []
    for start in ("greedy", "random"):
        for spec in SPECS:
            for algo in ALGOS:
                if _load_summary(start, spec["name"], algo["key"]) is None:
                    jobs.append((start, spec["name"], algo["key"]))
    return jobs


def _write_progress(pending, active):
    total = 2 * len(SPECS) * len(ALGOS)
    done = total - len(pending) - len(active)
    lines = [
        f"done {done}/{total}",
        f"waiting {len(pending)}",
        f"running {len(active)}",
        time.strftime("updated %Y-%m-%d %H:%M:%S"),
    ]
    for start, config_name, algo_key, proc, _handle in active:
        lines.append(f"  pid {proc.pid}  {start}  {config_name}  {algo_key}")
    (LOGS / "progress.txt").write_text("\n".join(lines) + "\n")


def _grid():
    pending = _pending()
    _log(f"Pending jobs: {len(pending)}  workers={WORKERS}  seconds={RUN_SECONDS}")
    active = []
    retried = set()
    while pending or active:
        while pending and len(active) < WORKERS:
            start, config_name, algo_key = pending.pop(0)
            log_path = LOGS / f"{start}__{config_name}__{algo_key}.log"
            handle = open(log_path, "w")
            proc = subprocess.Popen(
                [sys.executable, "-u", str(Path(__file__).resolve()), "job", start, config_name, algo_key],
                cwd=str(ROOT),
                stdout=handle,
                stderr=subprocess.STDOUT,
            )
            active.append((start, config_name, algo_key, proc, handle))
            _log(f"started {start} {config_name} {algo_key} pid={proc.pid}")
        _write_progress(pending, active)
        time.sleep(20)
        still = []
        finished = set()
        for item in active:
            start, config_name, algo_key, proc, handle = item
            code = proc.poll()
            if code is None:
                still.append(item)
                continue
            handle.close()
            key = (start, config_name, algo_key)
            if code != 0:
                if key not in retried:
                    retried.add(key)
                    pending.append(key)
                    _log(f"retry {start} {config_name} {algo_key} exit={code}")
                else:
                    _log(f"FAILED {start} {config_name} {algo_key} exit={code}")
            finished.add((start, config_name))
        active = still
        for start, config_name in finished:
            if plot_config(start, config_name):
                _log(f"plots {start} {config_name}")
    _log("Grid finished")


def _style(ax, title, xlabel, ylabel):
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    ax.legend(fontsize=8, ncol=2)


def _load_run(start, config_name, algo_key):
    path = _cache_path(start, config_name, algo_key)
    if not path.exists():
        return None
    with open(path, "rb") as handle:
        return pickle.load(handle)


def plot_config(start, config_name):
    runs = []
    for algo in ALGOS:
        run = _load_run(start, config_name, algo["key"])
        if run is None:
            return False
        run["label"] = algo["label"]
        run["algo_key"] = algo["key"]
        runs.append(run)
    spec = SPEC_BY_NAME[config_name]
    folder = OUT / start / config_name
    conv = folder / "convergence"
    folder.mkdir(parents=True, exist_ok=True)
    conv.mkdir(parents=True, exist_ok=True)
    initial = runs[0]["initial_cost"].total
    greedy_total = runs[0].get("greedy_total", initial)
    start_word = "greedy plan" if start == "greedy" else "random plan"

    fig, ax = plt.subplots(figsize=(11, 6.5))
    for run in runs:
        history = run["history"]
        ax.plot(
            _minutes(history),
            [row["best"] for row in history],
            color=COLORS[run["algo_key"]],
            linewidth=2,
            label=run["label"],
        )
    if start == "random":
        ax.axhline(greedy_total, color="#666666", linestyle=":", linewidth=1.2, label="Greedy plan")
    _style(ax, f"{_title(spec)} — best objective, {start_word}", "Minutes", "Best objective")
    fig.tight_layout()
    fig.savefig(folder / "cost_vs_time.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 6.5))
    for run in runs:
        history = run["history"]
        ax.plot(
            _minutes(history),
            [improvement(initial, row["best"]) for row in history],
            color=COLORS[run["algo_key"]],
            linewidth=2,
            label=run["label"],
        )
    _style(ax, f"{_title(spec)} — improvement vs {start_word}", "Minutes", "Improvement (%)")
    fig.tight_layout()
    fig.savefig(folder / "improvement_vs_time.png", dpi=140)
    plt.close(fig)

    names = ["Start"] + [run["label"] for run in runs]
    totals = [initial] + [run["best_cost"].total for run in runs]
    colors = ["#999999"] + [COLORS[run["algo_key"]] for run in runs]
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
    ax.set_title(f"{_title(spec)} — final objective, {start_word}")
    ax.set_ylabel("Objective value")
    ax.tick_params(axis="x", labelrotation=25)
    ax.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.7)
    fig.tight_layout()
    fig.savefig(folder / "total_cost.png", dpi=140)
    plt.close(fig)

    for run in runs:
        history = run["history"]
        fig, ax = plt.subplots(figsize=(10, 5.5))
        ax.plot(
            _minutes(history),
            [row["current"] for row in history],
            color="#56B4E9",
            linewidth=1,
            label="Current",
        )
        ax.plot(
            _minutes(history),
            [row["best"] for row in history],
            color=COLORS[run["algo_key"]],
            linewidth=2,
            label="Best",
        )
        ax.set_title(f"{_title(spec)} — {run['label']} — {start_word}")
        ax.set_xlabel("Minutes")
        ax.set_ylabel("Objective value")
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
        ax.legend()
        fig.tight_layout()
        fig.savefig(conv / f"{run['algo_key']}.png", dpi=140)
        plt.close(fig)
    return True


def plot_overview(start):
    rows = []
    for spec in SPECS:
        row = []
        for algo in ALGOS:
            summary = _load_summary(start, spec["name"], algo["key"])
            if summary is None:
                return False
            row.append(summary)
        rows.append(row)
    folder = OUT / start
    folder.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    x = list(range(len(SPECS)))
    for index, algo in enumerate(ALGOS):
        values = [rows[c][index]["improvement_vs_greedy"] for c in range(len(SPECS))]
        ax.plot(x, values, color=COLORS[algo["key"]], linewidth=2, marker="o", label=algo["label"])
    ax.set_xticks(x)
    ax.set_xticklabels([_title(spec) for spec in SPECS], rotation=20, ha="right")
    _style(ax, f"Improvement over the greedy plan — {start} start", "Instance", "Improvement (%)")
    fig.tight_layout()
    fig.savefig(folder / "improvement_by_config.png", dpi=140)
    plt.close(fig)
    return True


def _check():
    base, path = load_instance()
    print("instance", path)
    print(f"T75={T75:,.2f}  T25={T25:,.2f}  floor={T_FLOOR:,.2f}  sa50_floor={50.0 * RATIO:,.2f}")
    for spec in SPECS:
        data, meta = build_instance(base, spec)
        trips = build_initial_trips(data)
        cost, _aux = evaluate(Solution(data, clone_trips(trips)))
        print(
            f"{spec['name']:24} dealers={meta['dealers']:3} trucks={meta['trucks']:3} "
            f"demand={meta['demand_mode']} forecast={meta['forecast']:6} greedy={cost.total:,.0f}"
        )


def _acquire_lock():
    path = LOGS / "driver.pid"
    if path.exists():
        try:
            old = int(path.read_text().strip())
        except ValueError:
            old = None
        if old and old != os.getpid():
            try:
                os.kill(old, 0)
            except OSError:
                old = None
            else:
                raise SystemExit(f"driver already running, pid {old}")
    path.write_text(str(os.getpid()))


def main():
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    OUT.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    if len(sys.argv) >= 2 and sys.argv[1] == "check":
        _check()
        return
    if len(sys.argv) >= 2 and sys.argv[1] == "job":
        _run_job(sys.argv[2], sys.argv[3], sys.argv[4])
        return
    if len(sys.argv) >= 2 and sys.argv[1] == "plots":
        for start in ("greedy", "random"):
            for spec in SPECS:
                plot_config(start, spec["name"])
            plot_overview(start)
        return
    if len(sys.argv) >= 2 and sys.argv[1] == "doc":
        from analysis_report import write_results, write_speech
        write_results()
        write_speech()
        return
    _acquire_lock()
    run_calibration()
    _grid()
    for start in ("greedy", "random"):
        for spec in SPECS:
            plot_config(start, spec["name"])
        plot_overview(start)
    from analysis_report import write_results, write_speech
    write_results()
    write_speech()
    _log("Reports written")


if __name__ == "__main__":
    main()
