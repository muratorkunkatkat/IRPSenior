"""Final weights, then a 14-hour SA ladder on the greedy and random starts.

Step 0 is one hour of equal-weight SA at the config that won plots/test:
clock cooling from T0=15 to 15 * (paper T_final / T25). Offer weights are
1 / build success, capped at 5, scaled so the smallest is 1. Acceptance is
not used.

The ladder is seven one-hour runs on each start. 1e1, 1e2, 1e3 and 1e4 are
T0 = 15, 150, 1,500 and 15,000 with the same clock-cooling ratio. T25, T50
and T75 use the fast geometric alpha from the hour test: the schedule that
carries T50 to the paper floor in 20 minutes. Both starts use the lecturer
temperatures of the greedy plan.

    python test/sa_final_ladder.py
"""

import pickle
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "test") not in sys.path:
    sys.path.insert(0, str(ROOT / "test"))

from heuristic.search_problem import (
    MOVE_NAMES,
    Solution,
    build_initial_trips,
    build_random_trips,
    calibrated_weights,
    clone_trips,
    evaluate,
    move_stats,
    reset_move_stats,
    set_move_weights,
)
from heuristic.simulated_annealing import paper_temperatures, run_simulated_annealing
from calibrate_metaheuristics import (
    improvement,
    load_instance,
    print_cost,
    save_move_acceptance,
    save_sa_progress,
    save_variant_comparison,
)

SEED = 44
HOUR = 3600
# Same scored-neighbor rate the hour test used to set alpha.
SCORED_PER_SEC = 57886 / 3600.006352332
L = 10
FAST_DIVISOR = 3
ITERATIONS = 5_000_000
HISTORY_EVERY = 25
OUT = ROOT / "plots" / "final"
CACHE = OUT / "_cache"
COLORS = ["#0072B2", "#56B4E9", "#E69F00", "#F0E442", "#D55E00", "#CC79A7", "#009E73"]
DECADE = [("1e1", 15.0), ("1e2", 150.0), ("1e3", 1500.0), ("1e4", 15000.0)]


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


def _load(key):
    path = CACHE / f"{key}.pkl"
    if not path.exists():
        return None
    with open(path, "rb") as handle:
        return pickle.load(handle)


def _store(key, payload):
    CACHE.mkdir(parents=True, exist_ok=True)
    with open(CACHE / f"{key}.pkl", "wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)


def _sa(data, trips, t0, t_final, alpha):
    return run_simulated_annealing(
        data,
        iterations=ITERATIONS,
        seed=SEED,
        trips=trips,
        t0=t0,
        t_final=t_final,
        time_limit_sec=HOUR,
        log_every=20_000,
        alpha=alpha,
        cool_every=L,
        history_every=HISTORY_EVERY,
    )


def _report(label, initial_total, run):
    print_cost(f"{label} best", run["best_cost"])
    _log(
        f"  Improvement {improvement(initial_total, run['best_cost'].total):.2f}%  "
        f"best iter {run['best_iteration']} / {run['iterations_run']}  "
        f"end T={run['history'][-1]['temperature']:,.2f}  "
        f"current={run['history'][-1]['current']:,.2f}"
    )


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data, path = load_instance()
    greedy = build_initial_trips(data)
    greedy_cost, _aux = evaluate(Solution(data, clone_trips(greedy)))
    temps = paper_temperatures(greedy_cost.total)
    t25, t50, t75 = temps["T25"], temps["T50"], temps["T75"]
    paper_floor = temps["T_final"]
    ratio = paper_floor / t25
    t15_floor = 15.0 * ratio
    steps = SCORED_PER_SEC * HOUR / L
    alpha_fast = (paper_floor / t50) ** (FAST_DIVISOR / steps)

    _log(f"Instance: {path}")
    print_cost("Greedy start", greedy_cost)
    _log(
        f"Paper T75={t75:,.2f}  T50={t50:,.2f}  T25={t25:,.2f}  "
        f"T_final={paper_floor:,.2f}  ratio={ratio:.6f}"
    )
    _log(
        f"Fast alpha {alpha_fast:.10f}  "
        f"(T50 reaches {paper_floor:,.2f} at {HOUR / FAST_DIVISOR / 60:.0f} min "
        f"at {SCORED_PER_SEC:.2f} scored neighbors/s)"
    )

    cached = _load("s0")
    if cached is None:
        set_move_weights([1.0] * len(MOVE_NAMES))
        reset_move_stats()
        _log("")
        _log("=" * 60)
        _log(f"s0 equal weights  T0=15  floor={t15_floor:.4f}  clock  {HOUR:.0f}s")
        _log("=" * 60)
        run = _slim(_sa(data, greedy, 15.0, t15_floor, None))
        stats = move_stats()
        weights = calibrated_weights()
        _store("s0", {"run": run, "stats": stats, "weights": weights})
    else:
        _log("Cached s0")
        run = cached["run"]
        stats = cached["stats"]
        weights = cached["weights"]

    run["label"] = "s0 T15"
    _report("s0", greedy_cost.total, run)
    _log(f"{'move':<16} {'attempts':>9} {'built':>9} {'build%':>8} "
         f"{'accepted':>9} {'weight':>8}")
    for kind, weight in zip(MOVE_NAMES, weights):
        attempts = stats[kind]["attempts"]
        built = stats[kind]["built"]
        accepted = run["move_counts"].get(kind, 0)
        build_pct = 100.0 * built / attempts if attempts else 0.0
        _log(f"{kind:<16} {attempts:9d} {built:9d} {build_pct:7.1f}% "
             f"{accepted:9d} {weight:8.3f}")
    _log(f"Weights: {weights}")
    weight_dir = OUT / "weights"
    save_sa_progress(run, weight_dir, "Equal-weight SA at T0=15")
    save_move_acceptance(run, stats, weight_dir)
    _log(f"Weight plots: {weight_dir}")
    set_move_weights(weights)

    paper = [("T25", t25), ("T50", t50), ("T75", t75)]
    configs = []
    for label, t0 in DECADE:
        configs.append({
            "key": label,
            "label": label,
            "t0": t0,
            "t_final": t0 * ratio,
            "alpha": None,
        })
    for label, t0 in paper:
        configs.append({
            "key": f"{label}_fast",
            "label": f"{label} fast",
            "t0": t0,
            "t_final": paper_floor,
            "alpha": alpha_fast,
        })

    random_trips = build_random_trips(data, random.Random(SEED))
    random_cost, _aux = evaluate(Solution(data, clone_trips(random_trips)))
    _log("")
    print_cost("Random start", random_cost)

    for start_name, trips, initial, baseline in (
        ("greedy", greedy, greedy_cost, "Greedy"),
        ("random", random_trips, random_cost, "Random"),
    ):
        out_dir = OUT / start_name
        runs = []
        _log("")
        _log("#" * 60)
        _log(f"{start_name}  baseline {initial.total:,.2f}")
        _log("#" * 60)
        for cfg in configs:
            key = f"{start_name}_{cfg['key']}"
            _log("")
            _log("=" * 60)
            alpha_note = "clock" if cfg["alpha"] is None else f"alpha={cfg['alpha']:.10f}"
            _log(f"{start_name} {cfg['label']}  T0={cfg['t0']:,.2f}  "
                 f"floor={cfg['t_final']:,.4f}  {alpha_note}")
            _log("=" * 60)
            payload = _load(key)
            if payload is None:
                payload = _slim(_sa(data, trips, cfg["t0"], cfg["t_final"], cfg["alpha"]))
                _store(key, payload)
            else:
                _log(f"Cached {key}")
            payload["label"] = cfg["label"]
            runs.append(payload)
            _report(f"{start_name} {cfg['label']}", initial.total, payload)
            save_sa_progress(payload, out_dir / cfg["key"], f"SA {cfg['label']}")
            save_variant_comparison(
                runs, out_dir, COLORS[:len(runs)],
                f"SA {start_name} final weights", baseline,
            )
            _log(f"Plots updated: {out_dir}")
        best = min(runs, key=lambda item: item["best_cost"].total)
        _log(f"Best {start_name}: {best['label']} ({best['best_cost'].total:,.2f})")

    _log("Finished.")


if __name__ == "__main__":
    main()
