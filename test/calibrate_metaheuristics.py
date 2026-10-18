"""Step 0–2 calibration of SA temperatures and tabu neighborhood size.

Run from the project root:

    python test/calibrate_metaheuristics.py
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from heuristic.search_problem import (
    MOVE_NAMES,
    Solution,
    build_initial_trips,
    calibrated_weights,
    clone_trips,
    count_active,
    current_move_weights,
    evaluate,
    move_stats,
    reset_move_stats,
    set_move_weights,
)
from heuristic.simulated_annealing import paper_temperatures, run_simulated_annealing
from heuristic.tabu_search import run_tabu_search

TIME_LIMIT = 150
SEED = 44
ITERATIONS = 1_000_000
SA_COLOR = "#0072B2"
TS_COLOR = "#D55E00"
INIT_COLOR = "#009E73"
SA_VARIANT_COLORS = ["#0072B2", "#E69F00", "#009E73"]
TABU_VARIANT_COLORS = ["#D55E00", "#0072B2", "#009E73"]


def load_instance():
    path = ROOT / "data" / "data_exp.json"
    with open(path, "r") as handle:
        return json.load(handle), path


def print_cost(label, cost):
    print(f"{label}")
    print(f"  Transport              {cost.transport:15,.2f}")
    print(f"  Holding                {cost.holding:15,.2f}")
    print(f"  Backorder              {cost.backorder:15,.2f}   ({cost.backorder_units:,.0f} car-days)")
    print(f"  End-of-month unsold    {cost.unsold:15,.2f}   ({cost.unsold_units:,.0f} cars)")
    print(f"  Capacity penalty       {cost.capacity:15,.2f}")
    print(f"  Total                  {cost.total:15,.2f}")
    print(f"  Cars delivered         {cost.delivered:15,d}")


def improvement(initial, final):
    if initial == 0:
        return 0.0
    return 100.0 * (initial - final) / initial


def _series(history, key):
    return [row[key] for row in history]


def _style(ax, title, xlabel, ylabel):
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    ax.legend()


def save_sa_progress(sa, out_dir, title="Simulated annealing"):
    out_dir.mkdir(parents=True, exist_ok=True)
    history = sa["history"]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(_series(history, "seconds"), _series(history, "current"),
            color="#56B4E9", linewidth=1, label="Current")
    ax.plot(_series(history, "seconds"), _series(history, "best"),
            color=SA_COLOR, linewidth=2, label="Best")
    twin = ax.twinx()
    twin.plot(_series(history, "seconds"), _series(history, "temperature"),
              color="#CC79A7", linewidth=1, linestyle="--", label="Temperature")
    twin.set_ylabel("Temperature")
    ax.set_title(title)
    ax.set_xlabel("Seconds")
    ax.set_ylabel("Objective value")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = twin.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc="upper right")
    fig.tight_layout()
    fig.savefig(out_dir / "sa_convergence.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(_series(history, "seconds"), _series(history, "best_transport"),
            color=SA_COLOR, linewidth=2, label="Transport")
    ax.plot(_series(history, "seconds"), _series(history, "best_backorder"),
            color=TS_COLOR, linewidth=2, label="Backorder")
    ax.plot(_series(history, "seconds"), _series(history, "best_holding"),
            color=INIT_COLOR, linewidth=2, label="Holding")
    ax.plot(_series(history, "seconds"), _series(history, "best_unsold"),
            color="#CC79A7", linewidth=2, label="Unsold")
    _style(ax, f"{title} — best cost pieces", "Seconds", "Cost")
    fig.tight_layout()
    fig.savefig(out_dir / "sa_components_vs_time.png", dpi=140)
    plt.close(fig)


def save_move_acceptance(sa, stats, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    proposed = sa["proposed_counts"]
    accepted = sa["move_counts"]
    built_share = []
    accept_share = []
    labels = MOVE_NAMES
    for kind in labels:
        attempts = stats[kind]["attempts"] or 1
        built = stats[kind]["built"]
        prop = proposed.get(kind, 0) or 1
        built_share.append(100.0 * built / attempts)
        accept_share.append(100.0 * accepted.get(kind, 0) / prop)
    x = list(range(len(labels)))
    width = 0.38
    fig, ax = plt.subplots(figsize=(12, 5.8))
    ax.bar([i - width / 2 for i in x], built_share, width, color=INIT_COLOR,
           label="Built / attempted (offer success)")
    ax.bar([i + width / 2 for i in x], accept_share, width, color=SA_COLOR,
           label="Accepted / proposed (Metropolis)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    _style(ax, "Move offer success vs Metropolis acceptance", "Move", "Percent")
    fig.tight_layout()
    fig.savefig(out_dir / "move_acceptance.png", dpi=140)
    plt.close(fig)

    total_acc = sum(accepted.values()) or 1
    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.bar(labels, [100.0 * accepted.get(kind, 0) / total_acc for kind in labels],
           color=SA_COLOR)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_title("Share of accepted moves (equal proposal weights)")
    ax.set_ylabel("Share of accepted decisions (%)")
    ax.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.7)
    fig.tight_layout()
    fig.savefig(out_dir / "accepted_mix.png", dpi=140)
    plt.close(fig)


def save_variant_comparison(runs, out_dir, colors, title_prefix, baseline_name="Greedy"):
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    for run, color in zip(runs, colors):
        ax.plot(_series(run["history"], "seconds"), _series(run["history"], "best"),
                color=color, linewidth=2, label=run["label"])
    _style(ax, f"{title_prefix} — best objective vs time", "Seconds", "Best objective")
    fig.tight_layout()
    fig.savefig(out_dir / "cost_vs_time.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    initial = runs[0]["initial_cost"].total
    for run, color in zip(runs, colors):
        ax.plot(
            _series(run["history"], "seconds"),
            [improvement(initial, value) for value in _series(run["history"], "best")],
            color=color, linewidth=2, label=run["label"],
        )
    _style(ax, f"{title_prefix} — improvement vs {baseline_name.lower()}", "Seconds", "Improvement (%)")
    fig.tight_layout()
    fig.savefig(out_dir / "improvement_vs_time.png", dpi=140)
    plt.close(fig)

    names = [baseline_name] + [run["label"] for run in runs]
    costs = [runs[0]["initial_cost"]] + [run["best_cost"] for run in runs]
    bar_colors = [INIT_COLOR] + colors
    fig, ax = plt.subplots(figsize=(9, 5))
    totals = [cost.total for cost in costs]
    bars = ax.bar(names, totals, color=bar_colors)
    for bar, value in zip(bars, totals):
        ax.annotate(f"{value:,.0f}", xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    ha="center", va="bottom", fontsize=8)
    ax.set_title(f"{title_prefix} — final objective")
    ax.set_ylabel("Objective value")
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
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, (attr, panel) in zip(axes.ravel(), metrics):
        values = [getattr(cost, attr) for cost in costs]
        ax.bar(names, values, color=bar_colors)
        ax.set_title(panel)
        ax.tick_params(axis="x", labelrotation=20)
        ax.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.7)
    fig.suptitle(f"{title_prefix} — cost components")
    fig.tight_layout()
    fig.savefig(out_dir / "components.png", dpi=140)
    plt.close(fig)


def save_tabu_diagnostics(runs, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    labels = [run["label"] for run in runs]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5))
    axes[0].bar(labels, [run["forbidden_share"] for run in runs], color=TABU_VARIANT_COLORS)
    axes[0].set_title("Forbidden share (%)")
    axes[1].bar(labels, [run["worse_steps"] for run in runs], color=TABU_VARIANT_COLORS)
    axes[1].set_title("Worse steps")
    axes[2].bar(labels, [run["all_tabu_iterations"] for run in runs], color=TABU_VARIANT_COLORS)
    axes[2].set_title("All-tabu iterations")
    for ax in axes:
        ax.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.7)
        ax.tick_params(axis="x", labelrotation=15)
    fig.suptitle("Tabu memory diagnostics")
    fig.tight_layout()
    fig.savefig(out_dir / "diagnostics.png", dpi=140)
    plt.close(fig)


def main():
    data, path = load_instance()
    print("Instance:", path)
    print(f"Dealers {len(data['nodes_no_factory'])} | Trucks {len(data['K'])} | "
          f"Car types {len(data['C'])} | Days {len(data['T'])}")
    print(f"Time limit {TIME_LIMIT}s per run | seed {SEED}")
    print()

    base = build_initial_trips(data)
    initial, _aux = evaluate(Solution(data, clone_trips(base)))
    print(f"Dynamic greedy | active trips {count_active(base)}")
    print_cost("Starting plan", initial)
    temps = paper_temperatures(initial.total)
    print()
    print("Paper temperatures from this greedy total:")
    print(f"  reference {temps['reference']:,.2f}")
    print(f"  10% gap   {temps['delta10']:,.2f}")
    print(f"  T75 {temps['T75']:,.2f}   T50 {temps['T50']:,.2f}   T25 {temps['T25']:,.2f}")
    print(f"  T_final (1% gap kept at 0.0001) {temps['T_final']:,.2f}")
    print()

    set_move_weights([1.0] * len(MOVE_NAMES))
    reset_move_stats()
    print("=" * 60)
    print("STEP 0  SA T50, equal move weights")
    print("=" * 60)
    sa0 = run_simulated_annealing(
        data,
        iterations=ITERATIONS,
        seed=SEED,
        trips=base,
        t0=temps["T50"],
        t_final=temps["T_final"],
        time_limit_sec=TIME_LIMIT,
        log_every=1500,
    )
    print_cost("Step 0 best", sa0["best_cost"])
    print(f"  Improvement {improvement(initial.total, sa0['best_cost'].total):.2f}%")
    stats = move_stats()
    print()
    print(f"{'move':<16} {'attempts':>9} {'built':>9} {'build%':>8} "
          f"{'proposed':>9} {'accepted':>9} {'accept%':>8}")
    for kind in MOVE_NAMES:
        attempts = stats[kind]["attempts"]
        built = stats[kind]["built"]
        proposed = sa0["proposed_counts"].get(kind, 0)
        accepted = sa0["move_counts"].get(kind, 0)
        build_pct = 100.0 * built / attempts if attempts else 0.0
        acc_pct = 100.0 * accepted / proposed if proposed else 0.0
        print(f"{kind:<16} {attempts:9d} {built:9d} {build_pct:7.1f}% "
              f"{proposed:9d} {accepted:9d} {acc_pct:7.1f}%")
    weights = calibrated_weights()
    set_move_weights(weights)
    print()
    print("Calibrated weights (∝ 1 / build success, capped at 5×):")
    for kind, weight in zip(MOVE_NAMES, current_move_weights()):
        print(f"  {kind:<16} {weight:.3f}")
    print()
    initial_dir = ROOT / "plots" / "initial"
    save_sa_progress(sa0, initial_dir, "Step 0 SA (T50, equal weights)")
    save_move_acceptance(sa0, stats, initial_dir)
    print(f"Step 0 plots: {initial_dir}")
    print()

    sa_runs = []
    for label, key in (("T75", "T75"), ("T50", "T50"), ("T25", "T25")):
        reset_move_stats()
        print("=" * 60)
        print(f"STEP 1  SA {label}={temps[key]:,.2f}")
        print("=" * 60)
        run = run_simulated_annealing(
            data,
            iterations=ITERATIONS,
            seed=SEED,
            trips=base,
            t0=temps[key],
            t_final=temps["T_final"],
            time_limit_sec=TIME_LIMIT,
            log_every=1500,
        )
        run["label"] = label
        sa_runs.append(run)
        print_cost(f"{label} best", run["best_cost"])
        print(f"  Improvement {improvement(initial.total, run['best_cost'].total):.2f}%")
        print(f"  Last improvement at iteration {run['best_iteration']} / {run['iterations_run']}")
        print(f"  Accepted moves: {run['move_counts']}")
        print()
    sa_dir = ROOT / "plots" / "sa"
    save_variant_comparison(sa_runs, sa_dir, SA_VARIANT_COLORS, "SA temperature")
    print(f"Step 1 plots: {sa_dir}")
    best_sa = min(sa_runs, key=lambda run: run["best_cost"].total)
    print(f"Best SA temperature on this clock: {best_sa['label']} "
          f"({best_sa['best_cost'].total:,.2f})")
    print()

    tabu_runs = []
    for label, neighbors in (("A n=16", 16), ("B n=35", 35), ("C n=71", 71)):
        reset_move_stats()
        print("=" * 60)
        print(f"STEP 2  Tabu {label}, tabu size 12")
        print("=" * 60)
        run = run_tabu_search(
            data,
            iterations=ITERATIONS,
            neighbors=neighbors,
            tabu_size=12,
            seed=SEED,
            trips=base,
            time_limit_sec=TIME_LIMIT,
            log_every=25,
        )
        run["label"] = label
        tabu_runs.append(run)
        print_cost(f"{label} best", run["best_cost"])
        print(f"  Improvement {improvement(initial.total, run['best_cost'].total):.2f}%")
        print(f"  Iterations {run['iterations_run']}  last best {run['best_iteration']}")
        print(f"  Forbidden share {run['forbidden_share']:.3f}%  "
              f"({run['forbidden_candidates']}/{run['scored_candidates']})")
        print(f"  All-tabu iterations {run['all_tabu_iterations']}  "
              f"worse steps {run['worse_steps']}")
        print(f"  Accepted moves: {run['move_counts']}")
        print()
    tabu_dir = ROOT / "plots" / "tabu"
    save_variant_comparison(tabu_runs, tabu_dir, TABU_VARIANT_COLORS, "Tabu neighbors")
    save_tabu_diagnostics(tabu_runs, tabu_dir)
    print(f"Step 2 plots: {tabu_dir}")

    a_share = tabu_runs[0]["forbidden_share"]
    a_worse = tabu_runs[0]["worse_steps"]
    print()
    print("Tabu Pass 1 reading:")
    print(f"  Run A forbidden share {a_share:.3f}%, worse steps {a_worse}")
    live = False
    for run in tabu_runs[1:]:
        if run["forbidden_share"] >= 1.0 and run["worse_steps"] > 0:
            live = True
            print(f"  {run['label']}: list is live "
                  f"(forbidden {run['forbidden_share']:.2f}%, worse {run['worse_steps']})")
        else:
            print(f"  {run['label']}: forbidden {run['forbidden_share']:.3f}%, "
                  f"worse {run['worse_steps']}, all-tabu {run['all_tabu_iterations']}")
    slow = tabu_runs[-1]
    if slow["iterations_run"] < tabu_runs[0]["iterations_run"] / 4:
        print("  Sample of 71 looks too big for this clock (few iterations). Drop C for Pass 2.")
    if not live:
        print("  Neighbor count did not make the list matter. Next lever is a coarser "
              "tabu attribute, not a larger tabu size.")


if __name__ == "__main__":
    main()
