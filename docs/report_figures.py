"""Figures drawn only for the report: solution framework and move weights.

    python docs/report_figures.py
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "plots" / "report"
MOVE_LABELS = ["rebuild", "hand over", "exchange", "add car", "remove car", "add dealer", "remove dealer"]


def box(ax, x, y, w, h, text, edge, face, bold=False, size=11):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                                linewidth=2, edgecolor=edge, facecolor=face))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size,
            fontweight="bold" if bold else "normal", linespacing=1.35)


def arrow(ax, start, end, color="#333333", style="-|>", dashed=False):
    ax.annotate("", xy=end, xytext=start,
                arrowprops=dict(arrowstyle=style, color=color, linewidth=1.8,
                                linestyle="--" if dashed else "-", shrinkA=0, shrinkB=0))


def framework():
    fig, ax = plt.subplots(figsize=(11, 6.4))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 6.4)
    ax.axis("off")
    box(ax, 0.3, 5.15, 2.6, 0.95, "Problem data\n(dealers, trucks, demand,\ndistances, costs)", "#666666", "#f2f2f2")
    box(ax, 3.4, 5.15, 3.0, 0.95, "Starting plan\ngreedy plan  |  random plan", "#D55E00", "#fff3e6", bold=True)
    box(ax, 6.9, 5.15, 3.8, 0.95, "Objective function\ntransportation + holding + backorder\n+ unsold + capacity penalty",
        "#009E73", "#e6f4ee")
    box(ax, 2.6, 2.95, 4.6, 1.2, "Neighborhood (7 moves, weighted selection)\nrebuild, hand over, exchange, add car,\n"
        "remove car, add dealer, remove dealer\na move is rejected if a truck's tours overlap", "#0072B2", "#e8f1fb")
    box(ax, 0.7, 1.0, 4.35, 1.6, "Simulated annealing\none neighbor per iteration\naccepted with probability exp(−Δ/T)\n"
        "geometric cooling over time\nT0: T75, T25, 500, 50", "#0072B2", "#e8f1fb")
    box(ax, 5.95, 1.0, 4.35, 1.6, "Tabu search\n70 sampled neighbors per iteration\nbest allowed neighbor\n"
        "tabu attribute: the exact move\ntabu size 25 or 625, aspiration", "#332288", "#eeeaf6")
    box(ax, 3.9, 0.05, 3.2, 0.62, "Best plan within the time limit", "#666666", "#f2f2f2", bold=True)
    arrow(ax, (2.9, 5.62), (3.4, 5.62))
    arrow(ax, (6.4, 5.62), (6.9, 5.62))
    arrow(ax, (4.9, 5.15), (4.9, 4.15))
    arrow(ax, (3.9, 2.95), (2.9, 2.6))
    arrow(ax, (5.9, 2.95), (8.1, 2.6))
    arrow(ax, (2.9, 1.0), (4.6, 0.67))
    arrow(ax, (8.1, 1.0), (6.4, 0.67))
    arrow(ax, (8.8, 5.15), (7.2, 3.6), color="#009E73", style="<|-|>", dashed=True)
    ax.text(8.35, 4.25, "every neighbor is evaluated\nwith the objective function", color="#009E73", fontsize=9.5)
    fig.tight_layout()
    fig.savefig(OUT / "framework.png", dpi=200)
    plt.close(fig)


def weights():
    payload = json.loads((ROOT / "plots" / "analysis" / "weights.json").read_text())
    values = payload["weights"]
    total = sum(values)
    fig, ax = plt.subplots(figsize=(10, 5.8))
    bars = ax.bar(MOVE_LABELS, values, color="#0072B2")
    for bar, value in zip(bars, values):
        ax.annotate(f"{value:.2f}\n{100.0 * value / total:.0f}%",
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    ha="center", va="bottom", fontsize=9)
    ax.set_ylim(0, 5.8)
    ax.set_title("Move weights after the pilot run")
    ax.set_ylabel("Weight (1 to 5)")
    ax.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.7)
    fig.tight_layout()
    fig.savefig(OUT / "weights.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    framework()
    weights()
    print("saved", OUT / "framework.png", OUT / "weights.png")
