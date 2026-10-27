"""One-hour SA: two runtime alphas at T75/T50/T25, plus clock-cooled T0=15.

Alpha is the lecturer's geometric schedule, T <- alpha * T every L=10 scored
neighbors, floored at the paper T_final.

The standard alpha is the value that carries T50 down to T_final at 60 minutes,
using the scored-neighbor rate of a hot hour on this machine. The fast alpha is the value
that carries T50 down to T_final at 20 minutes. The same two numbers are used
at T75 and T25. The seventh run is clock cooling from T0=15, the setting that
won the earlier ladder.

    python test/sa_hour_alpha.py
"""

import math
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "test") not in sys.path:
    sys.path.insert(0, str(ROOT / "test"))

from heuristic.search_problem import (
    Solution,
    build_initial_trips,
    clone_trips,
    evaluate,
    set_move_weights,
)
from heuristic.simulated_annealing import paper_temperatures, run_simulated_annealing
from calibrate_metaheuristics import (
    improvement,
    load_instance,
    print_cost,
    save_sa_progress,
    save_variant_comparison,
)

# Step 0 at T0=1500: 1 / build success, cap 5, smallest = 1.
CALIBRATED_WEIGHTS = [5.0, 5.0, 5.0, 2.312, 5.0, 5.0, 1.0, 3.754]
SEED = 44
HOUR = 3600
# Cooling multiplies T every L scored neighbors, not every iteration. A hot
# hour on this instance scored 57,886 neighbors in 3,600 s (a discarded T75
# run whose alpha had been set from the iteration rate, so it stayed hot and
# the plan fell apart). A short probe on the clean greedy plan is several
# times faster and would leave the hour still hot.
SCORED_PER_SEC = 57886 / 3600.006352332
L = 10
# Fast schedule reaches the T50 floor at one third of the hour.
FAST_DIVISOR = 3
ITERATIONS = 5_000_000
HISTORY_EVERY = 25
OUT = ROOT / "plots" / "test"
CACHE = OUT / "_cache"
COLORS = ["#0072B2", "#56B4E9", "#E69F00", "#F0E442", "#D55E00", "#CC79A7", "#009E73"]


def _log(text):
    print(text, flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "run_log.txt", "a") as handle:
        handle.write(text + "\n")
        handle.flush()


def _slim(run):
    run.pop("accepted_flags", None)
    run.pop("best_trips", None)
    return run


def _load_or_run(key, builder):
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"{key}.pkl"
    if path.exists():
        _log(f"Cached {key}")
        with open(path, "rb") as handle:
            return pickle.load(handle)
    run = _slim(builder())
    with open(path, "wb") as handle:
        pickle.dump(run, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return run


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data, path = load_instance()
    set_move_weights(CALIBRATED_WEIGHTS)
    greedy = build_initial_trips(data)
    initial, _aux = evaluate(Solution(data, clone_trips(greedy)))
    temps = paper_temperatures(initial.total)
    t75, t50, t25 = temps["T75"], temps["T50"], temps["T25"]
    t_final = temps["T_final"]
    ratio = t_final / t25
    t15 = 15.0
    t15_final = t15 * ratio

    _log(f"Instance: {path}")
    _log(f"Weights: {CALIBRATED_WEIGHTS}")
    print_cost("Greedy start", initial)
    _log(
        f"Paper T75={t75:,.2f}  T50={t50:,.2f}  T25={t25:,.2f}  "
        f"T_final={t_final:,.2f}"
    )
    rate = SCORED_PER_SEC
    steps = rate * HOUR / L
    alpha_std = (t_final / t50) ** (1.0 / steps)
    alpha_fast = (t_final / t50) ** (FAST_DIVISOR / steps)
    _log(
        f"Scored-neighbor rate {rate:.2f}/s  "
        f"(57,886 scored in one hot hour)"
    )
    _log(
        f"Cooling steps in one hour at L={L}: {steps:,.0f}"
    )
    _log(
        f"Standard alpha {alpha_std:.10f}  "
        f"(T50 reaches {t_final:,.2f} at 60 min)"
    )
    _log(
        f"Fast alpha {alpha_fast:.10f}  "
        f"(T50 reaches {t_final:,.2f} at {HOUR / FAST_DIVISOR / 60:.0f} min)"
    )
    for name, start in (("T75", t75), ("T50", t50), ("T25", t25)):
        for tag, alpha in (("std", alpha_std), ("fast", alpha_fast)):
            cools = math.log(t_final / start) / math.log(alpha)
            minutes = cools * L / rate / 60.0
            _log(f"  {name} {tag}: floor at {minutes:.1f} min if the rate holds")
    _log(
        f"Seventh run: T0={t15:.0f} clock-cooled to {t15_final:.4f} over {HOUR:.0f}s"
    )
    _log("")

    configs = []
    for name, start in (("T75", t75), ("T50", t50), ("T25", t25)):
        for tag, alpha in (("std", alpha_std), ("fast", alpha_fast)):
            configs.append({
                "key": f"{name}_{tag}",
                "label": f"{name} {tag}",
                "t0": start,
                "t_final": t_final,
                "alpha": alpha,
            })
    configs.append({
        "key": "T15_clock",
        "label": "T15 clock",
        "t0": t15,
        "t_final": t15_final,
        "alpha": None,
    })

    runs = []
    for cfg in configs:
        _log("=" * 60)
        _log(f"{cfg['label']}  T0={cfg['t0']:,.2f}  floor={cfg['t_final']:,.4f}  "
             f"alpha={cfg['alpha']}")
        _log("=" * 60)

        def build(cfg=cfg):
            return run_simulated_annealing(
                data,
                iterations=ITERATIONS,
                seed=SEED,
                trips=greedy,
                t0=cfg["t0"],
                t_final=cfg["t_final"],
                time_limit_sec=HOUR,
                log_every=20_000,
                alpha=cfg["alpha"],
                cool_every=L,
                history_every=HISTORY_EVERY,
            )

        run = _load_or_run(cfg["key"], build)
        run["label"] = cfg["label"]
        runs.append(run)
        scored = sum(run["proposed_counts"].values())
        print_cost(f"{cfg['label']} best", run["best_cost"])
        _log(
            f"  Improvement {improvement(initial.total, run['best_cost'].total):.2f}%  "
            f"best iter {run['best_iteration']} / {run['iterations_run']}  "
            f"scored {scored} ({scored / run['elapsed']:.2f}/s)  "
            f"end T={run['history'][-1]['temperature']:,.2f}"
        )
        save_sa_progress(run, OUT / cfg["key"], f"SA {cfg['label']}")
        save_variant_comparison(
            runs, OUT, COLORS[:len(runs)], "SA one-hour comparison", "Greedy",
        )
        _log(f"Plots updated: {OUT}")
        _log("")

    best = min(runs, key=lambda item: item["best_cost"].total)
    _log(f"Best of the seven: {best['label']} ({best['best_cost'].total:,.2f})")
    _log("Finished.")


if __name__ == "__main__":
    main()
