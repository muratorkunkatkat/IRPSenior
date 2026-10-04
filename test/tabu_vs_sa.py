"""Compare simulated annealing and tabu search on data/data_exp.json.

Run from the project root:

    python test/tabu_vs_sa.py
"""

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from heuristic.search_problem import (
    Solution,
    build_initial_trips,
    clone_trips,
    count_active,
    evaluate,
)
from heuristic.simulated_annealing import run_simulated_annealing
from heuristic.tabu_search import run_tabu_search

# Budgets for the real-size file (71 dealers, 500 trucks, 5 types, 30 days).
# One tabu iteration scores TABU_NEIGHBORS solutions, so its wall time is longer.
SA_ITERATIONS = 1000
TABU_ITERATIONS = 200
TABU_NEIGHBORS = 16
TABU_TENURE = 12
SEED = 44
SA_TIME_LIMIT = 180
TABU_TIME_LIMIT = 240

SA_COLOR = "#0072B2"
TS_COLOR = "#D55E00"
INIT_COLOR = "#009E73"


def load_instance():
    path = ROOT / "data" / "data_exp.json"
    with open(path, "r") as handle:
        return json.load(handle), path


def improvement(initial, final):
    if initial == 0:
        return 0.0
    return 100.0 * (initial - final) / initial


def print_cost(label, cost):
    print(f"{label}")
    print(f"  Transport              {cost.transport:15,.2f}")
    print(f"  Holding                {cost.holding:15,.2f}")
    print(f"  Backorder              {cost.backorder:15,.2f}   ({cost.backorder_units:,.0f} car-days)")
    print(f"  End-of-month unsold    {cost.unsold:15,.2f}   ({cost.unsold_units:,.0f} cars)")
    print(f"  Capacity penalty       {cost.capacity:15,.2f}")
    print(f"  Total                  {cost.total:15,.2f}")
    print(f"  Cars delivered         {cost.delivered:15,d}")


def _series(history, key):
    return [row[key] for row in history]


def _style(ax, title, xlabel, ylabel):
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    ax.legend()


def _zoom_if_close(ax, values):
    lo = min(values)
    hi = max(values)
    if lo == 0 and hi == 0:
        ax.set_ylim(0, 1)
        return
    span = hi - lo
    if span / max(abs(hi), 1.0) < 0.2:
        pad = max(span * 0.8, abs(hi) * 0.002, 1.0)
        bottom = lo - pad
        if lo >= 0 and bottom < 0:
            bottom = 0
        ax.set_ylim(bottom, hi + pad * 1.4)
    else:
        ax.margins(y=0.15)


def save_comparison(sa, tabu, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(_series(sa["history"], "iteration"), _series(sa["history"], "best"),
            color=SA_COLOR, linewidth=2, label="Simulated annealing")
    ax.plot(_series(tabu["history"], "iteration"), _series(tabu["history"], "best"),
            color=TS_COLOR, linewidth=2, label="Tabu search")
    _style(ax, "Best objective by iteration", "Iteration", "Best objective value")
    fig.tight_layout()
    fig.savefig(out_dir / "cost_vs_iteration.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(_series(sa["history"], "seconds"), _series(sa["history"], "best"),
            color=SA_COLOR, linewidth=2, label="Simulated annealing")
    ax.plot(_series(tabu["history"], "seconds"), _series(tabu["history"], "best"),
            color=TS_COLOR, linewidth=2, label="Tabu search")
    _style(ax, "Best objective by wall-clock time", "Seconds", "Best objective value")
    fig.tight_layout()
    fig.savefig(out_dir / "cost_vs_time.png", dpi=140)
    plt.close(fig)

    initial_total = sa["initial_cost"].total
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(
        _series(sa["history"], "seconds"),
        [improvement(initial_total, value) for value in _series(sa["history"], "best")],
        color=SA_COLOR, linewidth=2, label="Simulated annealing",
    )
    ax.plot(
        _series(tabu["history"], "seconds"),
        [improvement(initial_total, value) for value in _series(tabu["history"], "best")],
        color=TS_COLOR, linewidth=2, label="Tabu search",
    )
    _style(ax, "Improvement over the dynamic greedy", "Seconds", "Improvement in objective (%)")
    fig.tight_layout()
    fig.savefig(out_dir / "improvement_vs_time.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(_series(sa["history"], "iteration"), _series(sa["history"], "best_transport"),
            color=SA_COLOR, linewidth=2, label="Simulated annealing")
    ax.plot(_series(tabu["history"], "iteration"), _series(tabu["history"], "best_transport"),
            color=TS_COLOR, linewidth=2, label="Tabu search")
    _style(ax, "Transport cost of the best solution", "Iteration", "Transport cost")
    fig.tight_layout()
    fig.savefig(out_dir / "transport_vs_iteration.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(_series(sa["history"], "iteration"), _series(sa["history"], "best_backorder"),
            color=SA_COLOR, linewidth=2, label="Simulated annealing")
    ax.plot(_series(tabu["history"], "iteration"), _series(tabu["history"], "best_backorder"),
            color=TS_COLOR, linewidth=2, label="Tabu search")
    _style(ax, "Backorder cost of the best solution", "Iteration", "Backorder cost")
    fig.tight_layout()
    fig.savefig(out_dir / "backorder_vs_iteration.png", dpi=140)
    plt.close(fig)

    metrics = [
        ("transport", "Transport"),
        ("holding", "Holding"),
        ("backorder", "Backorder"),
        ("unsold", "End-of-month unsold"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    names = ["Initial", "Simulated annealing", "Tabu search"]
    colors = [INIT_COLOR, SA_COLOR, TS_COLOR]
    costs = [sa["initial_cost"], sa["best_cost"], tabu["best_cost"]]
    for ax, (attr, title) in zip(axes.ravel(), metrics):
        values = [getattr(cost, attr) for cost in costs]
        bars = ax.bar(names, values, color=colors)
        ax.set_title(title)
        ax.tick_params(axis="x", labelrotation=15)
        ax.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.7)
        _zoom_if_close(ax, values)
        for bar, value in zip(bars, values):
            ax.annotate(f"{value:,.0f}", xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        ha="center", va="bottom", fontsize=8)
    fig.suptitle("Cost components")
    fig.tight_layout()
    fig.savefig(out_dir / "components.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    totals = [cost.total for cost in costs]
    bars = ax.bar(names, totals, color=colors)
    _zoom_if_close(ax, totals)
    for bar, value in zip(bars, totals):
        ax.annotate(f"{value:,.0f}", xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    ha="center", va="bottom", fontsize=9)
    ax.set_title("Final objective value")
    ax.set_ylabel("Objective value")
    ax.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.7)
    fig.tight_layout()
    fig.savefig(out_dir / "total_cost.png", dpi=140)
    plt.close(fig)

    kinds = sorted(set(sa["move_counts"]) | set(tabu["move_counts"]))
    fig, ax = plt.subplots(figsize=(10, 5))
    width = 0.38

    def share(counts):
        total = sum(counts.values()) or 1
        return [100.0 * counts.get(kind, 0) / total for kind in kinds]

    x = list(range(len(kinds)))
    ax.bar([i - width / 2 for i in x], share(sa["move_counts"]), width,
           color=SA_COLOR, label="Simulated annealing")
    ax.bar([i + width / 2 for i in x], share(tabu["move_counts"]), width,
           color=TS_COLOR, label="Tabu search")
    ax.set_xticks(x)
    ax.set_xticklabels(kinds, rotation=20)
    _style(ax, "Mix of accepted moves", "Move", "Share of accepted moves (%)")
    fig.tight_layout()
    fig.savefig(out_dir / "move_mix.png", dpi=140)
    plt.close(fig)


def save_sa_plots(sa, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    history = sa["history"]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(_series(history, "iteration"), _series(history, "current"),
            color="#9ECAE1", linewidth=1, label="Current")
    ax.plot(_series(history, "iteration"), _series(history, "best"),
            color=SA_COLOR, linewidth=2, label="Best")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Objective value")
    ax.set_title("Simulated annealing")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    twin = ax.twinx()
    twin.plot(_series(history, "iteration"), _series(history, "temperature"),
              color="#CC79A7", linewidth=1.2, linestyle="--", label="Temperature")
    twin.set_ylabel("Temperature")
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = twin.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc="upper right")
    fig.tight_layout()
    fig.savefig(out_dir / "convergence.png", dpi=140)
    plt.close(fig)

    flags = sa["accepted_flags"]
    if flags:
        window = 40
        rate = []
        running = 0
        for index, flag in enumerate(flags):
            running += int(flag)
            if index >= window:
                running -= int(flags[index - window])
                rate.append(running / window)
            else:
                rate.append(running / (index + 1))
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(range(1, len(rate) + 1), rate, color=SA_COLOR, linewidth=1.5)
        ax.set_ylim(0, 1)
        ax.set_title(f"Simulated annealing acceptance rate ({window}-iteration window)")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Fraction accepted")
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
        fig.tight_layout()
        fig.savefig(out_dir / "acceptance.png", dpi=140)
        plt.close(fig)


def save_tabu_plots(tabu, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    history = tabu["history"]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(_series(history, "iteration"), _series(history, "current"),
            color="#F4A261", linewidth=1, label="Current")
    ax.plot(_series(history, "iteration"), _series(history, "best"),
            color=TS_COLOR, linewidth=2, label="Best")
    _style(ax, "Tabu search", "Iteration", "Objective value")
    fig.tight_layout()
    fig.savefig(out_dir / "convergence.png", dpi=140)
    plt.close(fig)


def main():
    data, path = load_instance()
    dealers = len(data["nodes_no_factory"])
    print("Real-size instance:", path)
    print(f"Dealers {dealers} | Trucks {len(data['K'])} | "
          f"Car types {len(data['C'])} | Days {len(data['T'])}")
    demand = 0
    for dealer in data["nodes_no_factory"]:
        for car in data["C"]:
            demand += sum(data["F"][dealer][car])
    print(f"Forecasted demand over the month: {demand:,} cars")
    print()
    print("Objective = transport + daily holding + daily backorder")
    print("            + leftover cars after the last day (charged at G)")
    print("            + dealer storage overflow (charged at G per car-day).")
    print("Both algorithms start from the same dynamic-greedy solution.")
    print("One tabu iteration scores several neighbors, so equal iteration")
    print("counts are not equal work. cost_vs_time.png is the fair comparison.")
    print()

    built = time.perf_counter()
    base = build_initial_trips(data)
    build_sec = time.perf_counter() - built
    initial, _aux = evaluate(Solution(data, clone_trips(base)))
    print(f"Dynamic greedy built in {build_sec:.2f}s | active trips {count_active(base)}")
    print_cost("Initial solution", initial)
    print()

    sa = run_simulated_annealing(
        data,
        iterations=SA_ITERATIONS,
        seed=SEED,
        trips=base,
        time_limit_sec=SA_TIME_LIMIT,
    )
    print_cost("Simulated annealing best", sa["best_cost"])
    print(f"  Last improvement at iteration {sa['best_iteration']}")
    print(f"  Iterations run {sa['iterations_run']}  |  {sa['elapsed']:.1f}s  |  stop: {sa['stopped']}")
    print(f"  Accepted moves: {sa['move_counts']}")
    print(f"  Improvement vs initial: {improvement(initial.total, sa['best_cost'].total):.2f}%")
    if sa["best_iteration"] >= 0.85 * max(sa["iterations_run"], 1):
        print("  The best SA solution appeared late. A longer run may still improve it.")
    print()

    tabu = run_tabu_search(
        data,
        iterations=TABU_ITERATIONS,
        neighbors=TABU_NEIGHBORS,
        tenure=TABU_TENURE,
        seed=SEED,
        trips=base,
        time_limit_sec=TABU_TIME_LIMIT,
    )
    print_cost("Tabu search best", tabu["best_cost"])
    print(f"  Last improvement at iteration {tabu['best_iteration']}")
    print(f"  Iterations run {tabu['iterations_run']}  |  {tabu['elapsed']:.1f}s  |  stop: {tabu['stopped']}")
    print(f"  Accepted moves: {tabu['move_counts']}")
    print(f"  Improvement vs initial: {improvement(initial.total, tabu['best_cost'].total):.2f}%")
    if tabu["best_iteration"] >= 0.85 * max(tabu["iterations_run"], 1):
        print("  The best tabu solution appeared late. A longer run may still improve it.")
    print()

    gap = sa["best_cost"].total - tabu["best_cost"].total
    if abs(gap) < 0.05:
        winner = "Tie on total cost."
    elif gap > 0:
        winner = (f"Tabu search is lower by {gap:,.2f} "
                  f"({improvement(sa['best_cost'].total, tabu['best_cost'].total):.2f}% under SA).")
    else:
        winner = (f"Simulated annealing is lower by {-gap:,.2f} "
                  f"({improvement(tabu['best_cost'].total, sa['best_cost'].total):.2f}% under tabu).")
    print("Comparison:", winner)
    print(f"SA   {sa['elapsed']:.1f}s for {sa['iterations_run']} iterations "
          f"({sa['elapsed'] / max(sa['iterations_run'], 1):.3f}s each)")
    print(f"Tabu {tabu['elapsed']:.1f}s for {tabu['iterations_run']} iterations "
          f"({tabu['elapsed'] / max(tabu['iterations_run'], 1):.3f}s each)")

    comp_dir = ROOT / "plots" / "tabu_vs_sa"
    save_comparison(sa, tabu, comp_dir)
    save_sa_plots(sa, ROOT / "plots" / "sa")
    save_tabu_plots(tabu, ROOT / "plots" / "tabu")
    print()
    print(f"Comparison plots: {comp_dir}")
    print(f"SA plots:         {ROOT / 'plots' / 'sa'}")
    print(f"Tabu plots:       {ROOT / 'plots' / 'tabu'}")


if __name__ == "__main__":
    main()
