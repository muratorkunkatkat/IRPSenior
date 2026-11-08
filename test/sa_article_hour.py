"""Article temperatures on the nine-move neighborhood, greedy start only.

Duman & Duman (Evolutionary Intelligence, 2024), section 5:

  A neighbor that is 10% worse than the reference plan is accepted with
  probability 0.75, 0.50, or 0.25. Those three probabilities are the three
  initial temperatures. The final temperature is the one at which a neighbor
  1% worse than the same reference is accepted with probability 0.0001.
  That final temperature does not depend on which of the three starts was
  chosen. Alpha does. After every L scored neighbors the temperature is
  multiplied by alpha, and alpha is the multiplier that lands on the final
  temperature when the neighbor budget of this run is used up.

The neighbor budget here is one hour. L = 10, the value the article kept for
its smaller instances. The hour's neighbor count is measured on this move mix
before the alphas are fixed, and each initial temperature gets its own alpha.

The article has no second cooling rule. The fast schedule uses an alpha whose
exponent is three times larger, so the same hour ends at a colder temperature.
That colder value is the fast run's own final temperature.

One extra arm repeats the old winner: T0 = 15, clock cooling down to
15 * (article T_final / article T25) over the hour. Same moves, same equal
weights, same greedy start.
"""

import json
import pickle
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "test"))

import calibrate_metaheuristics as plots
from heuristic.search_problem import (
    MOVE_NAMES,
    Solution,
    build_initial_trips,
    evaluate,
    move_stats,
    reset_move_stats,
    set_move_weights,
)
from heuristic.simulated_annealing import paper_temperatures, run_simulated_annealing

SEED = 44
HOUR = 3600
L = 10
PROBE_SEC = 180
ITERATIONS = 5_000_000
HISTORY_EVERY = 25
LOG_EVERY = 10_000
WEIGHTS = [1.0] * len(MOVE_NAMES)

OUT = ROOT / "plots" / "test"
CACHE = OUT / "_cache_article"
LOG = OUT / "article_log.txt"

COLORS = ["#0072B2", "#56B4E9", "#D55E00", "#E69F00", "#009E73", "#CC79A7", "#000000"]


def _log(text):
    OUT.mkdir(parents=True, exist_ok=True)
    print(text, flush=True)
    with LOG.open("a", encoding="utf-8") as handle:
        handle.write(text + "\n")


def _alpha(t0, t_final, steps):
    if steps <= 0 or t0 <= t_final:
        return 1.0
    return (t_final / t0) ** (1.0 / steps)


def _slim(result):
    kept = dict(result)
    kept.pop("best_trips", None)
    kept.pop("accepted_flags", None)
    return kept


def _save_plots(runs):
    if not runs:
        return
    plots.save_variant_comparison(
        runs, OUT, COLORS[: len(runs)], "Article SA, greedy, equal moves", "Greedy",
    )
    for run in runs:
        plots.save_sa_progress(run, OUT / run["slug"], run["label"])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    data = json.loads((ROOT / "data" / "data_exp.json").read_text())
    greedy = build_initial_trips(data)
    reference = evaluate(Solution(data, greedy))[0].total
    temps = paper_temperatures(reference)
    ratio = temps["T_final"] / temps["T25"]
    t15_final = 15.0 * ratio

    _log("Article calculation on the greedy plan")
    _log(f"  reference C = {reference:,.6f}")
    _log("  acceptance: exp(-delta / T) = p, so T = -delta / ln(p)")
    _log(f"  delta_10 = 0.10 * C = {temps['delta10']:,.6f}")
    _log(f"  delta_1  = 0.01 * C = {temps['delta1']:,.6f}")
    _log(f"  T75 = -delta_10 / ln(0.75) = {temps['T75']:,.6f}")
    _log(f"  T50 = -delta_10 / ln(0.50) = {temps['T50']:,.6f}")
    _log(f"  T25 = -delta_10 / ln(0.25) = {temps['T25']:,.6f}")
    _log(f"  T_final = -delta_1 / ln(0.0001) = {temps['T_final']:,.6f}")
    _log("  T_final is the same number for every initial temperature.")
    _log("  It is fixed by the reference cost, not by which start was chosen.")
    _log(f"  old T0=15 clock floor = 15 * T_final / T25 = {t15_final:,.6f}")
    _log(f"  moves = {MOVE_NAMES}")
    _log(f"  weights = {WEIGHTS} so each move is drawn with probability 1/9")

    set_move_weights(WEIGHTS)
    reset_move_stats()
    probe = run_simulated_annealing(
        data, iterations=ITERATIONS, seed=SEED, trips=greedy,
        t0=temps["T50"], t_final=temps["T50"], time_limit_sec=PROBE_SEC,
        log_every=LOG_EVERY, alpha=None, cool_every=L, history_every=HISTORY_EVERY,
    )
    rate = probe["scored"] / probe["elapsed"] if probe["elapsed"] else 0.0
    steps = rate * HOUR / L
    _log(
        f"Probe at T50 for {probe['elapsed']:.1f}s: scored={probe['scored']} "
        f"iters={probe['iterations_run']} rate={rate:.4f} scored/s"
    )
    _log(f"Cooling steps in {HOUR}s at L={L}: N = rate * {HOUR} / {L} = {steps:.4f}")

    starts = [
        ("T75", temps["T75"]),
        ("T50", temps["T50"]),
        ("T25", temps["T25"]),
    ]
    jobs = []
    for name, t0 in starts:
        default_final = temps["T_final"]
        fast_final = t0 * (default_final / t0) ** 3
        jobs.append({
            "slug": f"{name}_default",
            "label": f"{name} default",
            "t0": t0,
            "t_final": default_final,
            "alpha": _alpha(t0, default_final, steps),
            "clock": False,
        })
        jobs.append({
            "slug": f"{name}_fast",
            "label": f"{name} fast",
            "t0": t0,
            "t_final": fast_final,
            "alpha": _alpha(t0, fast_final, steps),
            "clock": False,
        })
    jobs.append({
        "slug": "T15_clock",
        "label": "T0=15 clock",
        "t0": 15.0,
        "t_final": t15_final,
        "alpha": None,
        "clock": True,
    })

    for job in jobs:
        if job["clock"]:
            _log(
                f"{job['label']}: T0={job['t0']:.6f}  T_final={job['t_final']:.6f}  "
                "clock cooling over 3600s"
            )
        else:
            _log(
                f"{job['label']}: T0={job['t0']:.6f}  T_final={job['t_final']:.6f}  "
                f"alpha={job['alpha']:.12f}  L={L}"
            )

    runs = []
    for job in jobs:
        path = CACHE / f"{job['slug']}.pkl"
        if path.exists():
            with path.open("rb") as handle:
                result = pickle.load(handle)
            _log(f"cached {job['label']} best={result['best_cost'].total:,.2f}")
        else:
            set_move_weights(WEIGHTS)
            reset_move_stats()
            started = time.strftime("%Y-%m-%d %H:%M:%S")
            _log(f"START {job['label']} at {started}")
            result = run_simulated_annealing(
                data, iterations=ITERATIONS, seed=SEED, trips=greedy,
                t0=job["t0"], t_final=job["t_final"], time_limit_sec=HOUR,
                log_every=LOG_EVERY, alpha=job["alpha"], cool_every=L,
                history_every=HISTORY_EVERY,
            )
            result["label"] = job["label"]
            result["slug"] = job["slug"]
            result["move_stats"] = move_stats()
            best = result["best_cost"]
            _log(
                f"DONE {job['label']} best={best.total:,.2f} "
                f"transport={best.transport:,.2f} holding={best.holding:,.2f} "
                f"backorder={best.backorder:,.2f} unsold={best.unsold:,.2f} "
                f"iter={result['best_iteration']}/{result['iterations_run']} "
                f"scored={result['scored']} T_end={result['history'][-1]['temperature']:.6f} "
                f"seconds={result['elapsed']:.1f}"
            )
            with path.open("wb") as handle:
                pickle.dump(_slim(result), handle)
        result["label"] = job["label"]
        result["slug"] = job["slug"]
        runs.append(result)
        _save_plots(runs)

    _log("all article-move hours finished")


if __name__ == "__main__":
    main()
