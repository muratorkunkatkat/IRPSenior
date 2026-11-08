"""Reweighted 30-minute study: 5 SA settings and 4 tabu settings on 10 instances.

Weights come from one 30-minute SA on the random start at T25, equal weights,
using the usual inverse build-rate calibration. Searches then use those weights.

    python test/analysis_grid.py check
    python test/analysis_grid.py            # calibrate if needed, then the grid, then the doc
    python test/analysis_grid.py job NAME ALGO
    python test/analysis_grid.py plots
    python test/analysis_grid.py doc
"""

import json
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

from data.generate_json_data import (
    CITIES,
    COST_PER_KM,
    DAILY_MEAN,
    FACTORY,
    ROAD_FACTOR,
    dealer_sites,
    demand_pool,
    haversine_km,
    travel_days,
)
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
from heuristic.simulated_annealing import run_simulated_annealing
from heuristic.tabu_search import run_tabu_search
from calibrate_metaheuristics import (
    improvement,
    load_instance,
    print_cost,
    save_move_acceptance,
    save_sa_progress,
)

SEED = 44
HALF = 1800
WORKERS = 4
SA_ITERATIONS = 4_000_000
TABU_ITERATIONS = 1_000_000
HISTORY_EVERY = 25
DELTA = 6685.0
T75 = -DELTA / math.log(0.75)
T50 = -DELTA / math.log(0.50)
T25 = -DELTA / math.log(0.25)
T_FLOOR = -(DELTA / 10.0) / math.log(0.0001)
RATIO = T_FLOOR / T25

OUT = ROOT / "plots" / "analysis"
CACHE = OUT / "_cache"
INITIAL = ROOT / "plots" / "initial"
WEIGHTS_PATH = OUT / "weights.json"
LOG = OUT / "log.txt"

SA_JOBS = (
    ("T75", T75, T_FLOOR),
    ("T50", T50, T_FLOOR),
    ("T25", T25, T_FLOOR),
    ("500", 500.0, T_FLOOR),
    ("50", 50.0, 50.0 * RATIO),
)
TABU_JOBS = (
    (35, 25),
    (35, 625),
    (70, 25),
    (70, 625),
)

# Extra sites on cities that already exist. 39 of them, so the instance has 110 dealers.
EXTRA_SITES = (
    ("İstanbul", 12),
    ("Ankara", 6),
    ("İzmir", 4),
    ("Bursa", 3),
    ("Antalya", 3),
    ("Adana", 2),
    ("Konya", 2),
    ("Gaziantep", 2),
    ("Kocaeli", 2),
    ("Mersin", 2),
    ("Kayseri", 1),
)

CONFIGS = (
    {
        "name": "00_original",
        "title": "Orijinal",
        "dealers": "all",
        "trucks": 80,
        "demand": 1.0,
        "note": "71 bayi, 80 tır, talep 1×. Karşılaştırmanın tabanı.",
    },
    {
        "name": "01_scaled",
        "title": "Büyümüş firma",
        "dealers": "large",
        "trucks": 120,
        "demand": 1.0,
        "note": "110 bayi ve 120 tır. Talep yoğunluğu aynı. Firma ölçeklenmiş.",
    },
    {
        "name": "02_growth_no_fleet",
        "title": "Büyüme, filo aynı",
        "dealers": "large",
        "trucks": 80,
        "demand": 2.0,
        "note": "110 bayi, talep 2×, filo 80'de kalmış.",
    },
    {
        "name": "03_large_tight",
        "title": "Büyük harita, dar filo",
        "dealers": "large",
        "trucks": 30,
        "demand": 1.0,
        "note": "110 bayi, 30 tır, talep 1×.",
    },
    {
        "name": "04_near_tight_heavy",
        "title": "Yakın bölge, sıkışık",
        "dealers": "near",
        "trucks": 30,
        "demand": 2.0,
        "note": "Fabrikaya aynı gün varılan 23 bayi, 30 tır, talep 2×.",
    },
    {
        "name": "05_sparse_slack",
        "title": "Seyrek ülke, rahat filo",
        "dealers": "national",
        "trucks": 140,
        "demand": 0.5,
        "note": "24 şehirlik ülke kesiti, 140 tır, talep 0,5×.",
    },
    {
        "name": "06_full_tight_heavy",
        "title": "Aynı harita, yüklenmiş",
        "dealers": "all",
        "trucks": 30,
        "demand": 2.0,
        "note": "71 bayi duruyor. Filo 30, talep 2×.",
    },
    {
        "name": "07_full_slack_thin",
        "title": "Aynı harita, boşalmış",
        "dealers": "all",
        "trucks": 140,
        "demand": 0.5,
        "note": "71 bayi duruyor. Filo 140, talep 0,5×.",
    },
    {
        "name": "08_near_thin",
        "title": "Yakın bölge, seyrek talep",
        "dealers": "near",
        "trucks": 80,
        "demand": 0.5,
        "note": "23 yakın bayi, 80 tır, talep 0,5×.",
    },
    {
        "name": "09_far_tight",
        "title": "Uzak iskelet, dar filo",
        "dealers": "national",
        "trucks": 30,
        "demand": 1.0,
        "note": "24 uzak-yakın karışık şehir, 30 tır, talep 1×.",
    },
)


def _log(text):
    print(text, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG, "a") as handle:
        handle.write(text + "\n")
        handle.flush()


def _algo_list():
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


ALGOS = _algo_list()
ALGO_BY_KEY = {item["key"]: item for item in ALGOS}
COLORS = {
    "sa_T75": "#0072B2",
    "sa_T50": "#E69F00",
    "sa_T25": "#009E73",
    "sa_500": "#D55E00",
    "sa_50": "#CC79A7",
    "tabu_n35_t25": "#56B4E9",
    "tabu_n35_t625": "#882255",
    "tabu_n70_t25": "#332288",
    "tabu_n70_t625": "#000000",
}


def _city_table():
    rows = []
    index = 1
    for name, lat, lon, count, yard in CITIES:
        rows.append({
            "name": name,
            "lat": lat,
            "lon": lon,
            "count": count,
            "yard": yard,
            "first": index,
        })
        index += count
    return rows


def near_dealer_ids(data):
    return [dealer for dealer in data["nodes_no_factory"] if data["A"][0][dealer] == 0]


def national_dealer_ids():
    cities = _city_table()
    multi = [city for city in cities if city["count"] > 1]
    singles = sorted((city for city in cities if city["count"] == 1), key=lambda city: city["lon"])
    chosen = {city["name"] for city in multi}
    ids = [city["first"] for city in multi]
    slots = 24 - len(ids)
    for step in range(slots):
        position = round(step * (len(singles) - 1) / max(slots - 1, 1))
        city = singles[position]
        if city["name"] in chosen:
            continue
        chosen.add(city["name"])
        ids.append(city["first"])
    if len(ids) < 24:
        for city in singles:
            if city["name"] in chosen:
                continue
            ids.append(city["first"])
            chosen.add(city["name"])
            if len(ids) == 24:
                break
    return sorted(ids[:24])


def _scale_series(series, mode, node, car):
    if mode == 1.0:
        return list(series)
    if mode == 2.0:
        return [int(value) * 2 for value in series]
    scaled = []
    for day, value in enumerate(series):
        if value >= 2:
            scaled.append(int(value) // 2)
        elif value == 1 and (node + car + day) % 2 == 0:
            scaled.append(0)
        else:
            scaled.append(int(value))
    return scaled


def _blank_demand(n_nodes, n_types, n_days):
    return [[[0] * n_days for _ in range(n_types)] for _ in range(n_nodes)]


def _extra_dealers():
    by_name = {name: (lat, lon, yard) for name, lat, lon, _count, yard in CITIES}
    added = []
    for name, count in EXTRA_SITES:
        lat, lon, yard = by_name[name]
        for step in range(count):
            angle = 2.0 * math.pi * (step + 0.5) / count + 0.4
            radius = 0.11 + 0.015 * (step % 3)
            added.append((name, lat + radius * math.cos(angle), lon + radius * 1.25 * math.sin(angle), yard))
    if len(added) != 39:
        raise RuntimeError(f"expected 39 extra dealers, built {len(added)}")
    return added


def _pair_cost(a, b):
    km = haversine_km(a[0], a[1], b[0], b[1]) * ROAD_FACTOR
    return travel_days(km), round(km * COST_PER_KM, 2)


def _submatrix(matrix, ids):
    return [[matrix[i][j] for j in ids] for i in ids]


def build_instance(base, spec):
    if spec["dealers"] == "all":
        dealer_ids = list(base["nodes_no_factory"])
        extras = []
    elif spec["dealers"] == "near":
        dealer_ids = near_dealer_ids(base)
        extras = []
    elif spec["dealers"] == "national":
        dealer_ids = national_dealer_ids()
        extras = []
    elif spec["dealers"] == "large":
        dealer_ids = list(base["nodes_no_factory"])
        extras = _extra_dealers()
    else:
        raise ValueError(spec["dealers"])

    old_ids = [0] + dealer_ids
    n_days = len(base["T"])
    n_types = len(base["C"])
    n_nodes = len(old_ids) + len(extras)
    demand_mode = spec["demand"]

    demand = _blank_demand(n_nodes, n_types, n_days)
    for new_index, old_index in enumerate(old_ids):
        for car in range(n_types):
            demand[new_index][car] = _scale_series(
                base["F"][old_index][car], demand_mode, new_index, car
            )

    if extras:
        rng = random.Random(SEED + 110)
        cursor = {car: 0 for car in range(n_types)}
        pools = {}
        for car, mean in enumerate(DAILY_MEAN):
            pool, _probs = demand_pool(mean, len(extras) * n_days, rng)
            pools[car] = pool
        for offset, _extra in enumerate(extras):
            new_index = len(old_ids) + offset
            for car in range(n_types):
                raw = []
                for _day in range(n_days):
                    raw.append(pools[car][cursor[car]])
                    cursor[car] += 1
                demand[new_index][car] = _scale_series(raw, demand_mode, new_index, car)

    sites = dealer_sites()
    coords = [FACTORY] + [(lat, lon) for _name, lat, lon, _yard in sites]
    coords = [coords[index] for index in old_ids]
    yards = [base["V"][index] for index in old_ids]
    for name, lat, lon, yard in extras:
        coords.append((lat, lon))
        yards.append(yard)

    travel = _submatrix(base["A"], old_ids)
    cost = _submatrix(base["R"], old_ids)
    for row in travel:
        row.extend([0] * len(extras))
    for row in cost:
        row.extend([0.0] * len(extras))
    for _extra in extras:
        travel.append([0] * n_nodes)
        cost.append([0.0] * n_nodes)
    base_n = len(old_ids)
    for new_index in range(base_n, n_nodes):
        for other in range(new_index):
            days, fare = _pair_cost(coords[new_index], coords[other])
            travel[new_index][other] = travel[other][new_index] = days
            cost[new_index][other] = cost[other][new_index] = fare

    forecast = 0
    for dealer in range(1, n_nodes):
        for car in range(n_types):
            forecast += sum(demand[dealer][car])

    data = {
        "C": list(base["C"]),
        "K": list(range(spec["trucks"])),
        "N": list(range(n_nodes)),
        "T": list(base["T"]),
        "nodes_no_factory": list(range(1, n_nodes)),
        "A": travel,
        "R": cost,
        "Q": list(base["Q"]),
        "V": yards,
        "B": list(base["B"]),
        "H": list(base["H"]),
        "G": list(base["G"]),
        "P": list(base["P"]),
        "F": demand,
    }
    meta = {
        "name": spec["name"],
        "title": spec["title"],
        "note": spec["note"],
        "dealers": n_nodes - 1,
        "trucks": spec["trucks"],
        "demand_mode": demand_mode,
        "forecast": forecast,
        "dealer_kind": spec["dealers"],
    }
    return data, meta


def _cache_path(config_name, algo_key):
    return CACHE / f"{config_name}__{algo_key}.pkl"


def _summary_path(config_name, algo_key):
    return CACHE / f"{config_name}__{algo_key}.json"


def _slim(run):
    run.pop("best_trips", None)
    run.pop("accepted_flags", None)
    return run


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


def _summarize(run, initial):
    history = run["history"]
    best = run["best_cost"]
    row = {
        "best": best.total,
        "initial": initial.total,
        "improvement": improvement(initial.total, best.total),
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
    for second in (600, 1200, 1800):
        mark = _snapshot(history, second)
        row[f"best_at_{second}"] = mark["best"]
        row[f"current_at_{second}"] = mark["current"]
    return row


def _apply_weights():
    with open(WEIGHTS_PATH) as handle:
        payload = json.load(handle)
    set_move_weights(payload["weights"])
    return payload["weights"]


def _run_job(config_name, algo_key):
    spec = next(item for item in CONFIGS if item["name"] == config_name)
    algo = ALGO_BY_KEY[algo_key]
    base, _path = load_instance()
    data, meta = build_instance(base, spec)
    _apply_weights()
    reset_move_stats()
    trips = build_initial_trips(data)
    initial, _aux = evaluate(Solution(data, clone_trips(trips)))
    if algo["family"] == "sa":
        run = run_simulated_annealing(
            data,
            iterations=SA_ITERATIONS,
            seed=SEED,
            trips=trips,
            t0=algo["t0"],
            t_final=algo["floor"],
            time_limit_sec=HALF,
            log_every=20_000,
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
            time_limit_sec=HALF,
            log_every=1_000,
        )
    run["label"] = algo["label"]
    run["algo_key"] = algo_key
    run["config"] = config_name
    summary = _summarize(run, initial)
    summary["label"] = algo["label"]
    summary["algo_key"] = algo_key
    summary["family"] = algo["family"]
    summary["config"] = config_name
    summary["greedy"] = _cost_dict(initial)
    summary["initial_tours"] = len(trips)
    summary["meta"] = meta
    if algo["family"] == "sa":
        summary["t0"] = algo["t0"]
        summary["floor"] = algo["floor"]
    else:
        summary["neighbors"] = algo["neighbors"]
        summary["tenure"] = algo["tenure"]
    CACHE.mkdir(parents=True, exist_ok=True)
    with open(_cache_path(config_name, algo_key), "wb") as handle:
        pickle.dump(_slim(run), handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(_summary_path(config_name, algo_key), "w") as handle:
        json.dump(summary, handle)
    _log(
        f"done {config_name} {algo['label']}  "
        f"best={summary['best']:,.2f}  imp={summary['improvement']:.2f}%  "
        f"iter={summary['best_iteration']}/{summary['iterations']}"
    )


def _load_run(config_name, algo_key):
    path = _cache_path(config_name, algo_key)
    if not path.exists():
        return None
    with open(path, "rb") as handle:
        return pickle.load(handle)


def _load_summary(config_name, algo_key):
    path = _summary_path(config_name, algo_key)
    if not path.exists():
        return None
    with open(path) as handle:
        return json.load(handle)


def _style(ax, title, xlabel, ylabel):
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    ax.legend(fontsize=8, ncol=2)


def plot_config(config_name):
    runs = []
    for algo in ALGOS:
        run = _load_run(config_name, algo["key"])
        if run is None:
            return False
        run["label"] = algo["label"]
        run["algo_key"] = algo["key"]
        runs.append(run)
    spec = next(item for item in CONFIGS if item["name"] == config_name)
    folder = OUT / config_name
    folder.mkdir(parents=True, exist_ok=True)
    initial = runs[0]["initial_cost"].total

    fig, ax = plt.subplots(figsize=(11, 6.5))
    for run in runs:
        history = run["history"]
        ax.plot(
            [row["seconds"] for row in history],
            [row["best"] for row in history],
            color=COLORS[run["algo_key"]],
            linewidth=2,
            label=run["label"],
        )
    _style(ax, f"{spec['title']} — best objective vs time", "Seconds", "Best objective")
    fig.tight_layout()
    fig.savefig(folder / "cost_vs_time.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 6.5))
    for run in runs:
        history = run["history"]
        ax.plot(
            [row["seconds"] for row in history],
            [improvement(initial, row["best"]) for row in history],
            color=COLORS[run["algo_key"]],
            linewidth=2,
            label=run["label"],
        )
    _style(ax, f"{spec['title']} — improvement vs greedy", "Seconds", "Improvement (%)")
    fig.tight_layout()
    fig.savefig(folder / "improvement_vs_time.png", dpi=140)
    plt.close(fig)

    names = ["Greedy"] + [run["label"] for run in runs]
    totals = [initial] + [run["best_cost"].total for run in runs]
    colors = ["#999999"] + [COLORS[run["algo_key"]] for run in runs]
    fig, ax = plt.subplots(figsize=(12, 6))
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
    ax.set_title(f"{spec['title']} — final objective")
    ax.set_ylabel("Objective value")
    ax.tick_params(axis="x", labelrotation=30)
    ax.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.7)
    fig.tight_layout()
    fig.savefig(folder / "total_cost.png", dpi=140)
    plt.close(fig)
    return True


def plot_overview():
    summaries = []
    for spec in CONFIGS:
        row = []
        for algo in ALGOS:
            summary = _load_summary(spec["name"], algo["key"])
            if summary is None:
                return
            row.append(summary)
        summaries.append(row)
    folder = OUT / "overview"
    folder.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 6.5))
    x = list(range(len(CONFIGS)))
    for index, algo in enumerate(ALGOS):
        values = [summaries[c][index]["improvement"] for c in range(len(CONFIGS))]
        ax.plot(x, values, color=COLORS[algo["key"]], linewidth=2, marker="o", label=algo["label"])
    ax.set_xticks(x)
    ax.set_xticklabels([spec["title"] for spec in CONFIGS], rotation=30, ha="right")
    _style(ax, "Improvement over greedy, all configs", "Config", "Improvement (%)")
    fig.tight_layout()
    fig.savefig(folder / "improvement_by_config.png", dpi=140)
    plt.close(fig)


def _reweight_alive():
    probe = subprocess.run(["pgrep", "-f", "test/reweight_t25.py"], capture_output=True)
    return probe.returncode == 0


def _copy_weight_file(src):
    payload = json.loads(src.read_text())
    stored = {
        "weights": payload["weights"],
        "names": payload.get("moves", payload.get("names", list(MOVE_NAMES))),
        "stats": payload.get("stats", {}),
        "move_counts": payload.get("move_counts", {}),
        "proposed_counts": payload.get("proposed_counts", {}),
        "t0": payload.get("T0", payload.get("t0", T25)),
        "floor": payload.get("floor", T_FLOOR),
        "seed": payload.get("seed", SEED),
        "seconds": payload.get("seconds", HALF),
        "start": payload.get("start", "random"),
        "initial_total": payload.get("initial", payload.get("initial_total")),
        "best_total": payload.get("best", payload.get("best_total")),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    with open(WEIGHTS_PATH, "w") as handle:
        json.dump(stored, handle, indent=2)
    _log("Weights: " + ", ".join(
        f"{name}={weight:.3f}" for name, weight in zip(stored["names"], stored["weights"])
    ))


def ensure_weights():
    if WEIGHTS_PATH.exists():
        _log(f"Weights already at {WEIGHTS_PATH}")
        return
    src = INITIAL / "weights.json"
    deadline = time.time() + 50 * 60
    _log("Waiting for the T25 random-start calibration to write plots/initial/weights.json")
    while time.time() < deadline:
        if src.exists():
            _copy_weight_file(src)
            return
        if not _reweight_alive():
            time.sleep(3)
            if src.exists():
                _copy_weight_file(src)
                return
            _log("Calibration process ended without a weight file. Running it once.")
            run_calibration()
            return
        time.sleep(20)
    raise SystemExit("Calibration did not finish within 50 minutes")


def run_calibration():
    if WEIGHTS_PATH.exists():
        _log(f"Weights already at {WEIGHTS_PATH}")
        return
    data, path = load_instance()
    set_move_weights([1.0] * len(MOVE_NAMES))
    reset_move_stats()
    rng = random.Random(SEED)
    trips = build_random_trips(data, rng)
    initial, _aux = evaluate(Solution(data, clone_trips(trips)))
    _log(f"Calibration instance: {path}")
    _log(f"Random start, seed {SEED}. Equal weights. T0={T25:,.2f}  floor={T_FLOOR:.2f}  {HALF}s")
    print_cost("Random start", initial)
    run = run_simulated_annealing(
        data,
        iterations=SA_ITERATIONS,
        seed=SEED,
        trips=trips,
        t0=T25,
        t_final=T_FLOOR,
        time_limit_sec=HALF,
        log_every=20_000,
        alpha=None,
        history_every=HISTORY_EVERY,
    )
    print_cost("Calibration best", run["best_cost"])
    stats = move_stats()
    weights = calibrated_weights()
    _store_weights(weights, stats, run, initial)
    _write_initial_plots(run, stats, weights)


def _store_weights(weights, stats, run, initial):
    payload = {
        "weights": weights,
        "names": list(MOVE_NAMES),
        "stats": stats,
        "move_counts": run.get("move_counts", {}),
        "proposed_counts": run.get("proposed_counts", {}),
        "t0": T25,
        "floor": T_FLOOR,
        "seed": SEED,
        "seconds": HALF,
        "start": "random",
        "initial_total": initial.total,
        "best_total": run["best_cost"].total,
        "improvement": improvement(initial.total, run["best_cost"].total),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    with open(WEIGHTS_PATH, "w") as handle:
        json.dump(payload, handle, indent=2)
    _log("Weights: " + ", ".join(f"{name}={weight:.3f}" for name, weight in zip(MOVE_NAMES, weights)))


def _write_initial_plots(run, stats, weights):
    INITIAL.mkdir(parents=True, exist_ok=True)
    save_sa_progress(run, INITIAL, "Equal-weight SA, random start, T25, 30 min")
    save_move_acceptance(run, stats, INITIAL)
    with open(INITIAL / "weights.txt", "w") as handle:
        handle.write("source: random start, seed 44, T0=4822.21, floor=72.58, 1800s, equal weights\n")
        handle.write("rule: weight = min(5, (1/build_rate) / min(1/build_rate))\n")
        for name, weight in zip(MOVE_NAMES, weights):
            attempts = stats[name]["attempts"]
            built = stats[name]["built"]
            rate = built / attempts if attempts else 0.0
            handle.write(f"{name}\t{weight:.6f}\tattempts={attempts}\tbuilt={built}\trate={rate:.6f}\n")
    _log(f"Replaced plots in {INITIAL}")


def _write_progress(pending, active):
    done = 90 - len(pending) - len(active)
    lines = [f"done {done}/90", f"waiting {len(pending)}", f"running {len(active)}"]
    for config_name, algo_key, proc, _handle in active:
        lines.append(f"  pid {proc.pid}  {config_name}  {algo_key}")
    (OUT / "progress.txt").write_text("\n".join(lines) + "\n")


def _refresh_doc():
    try:
        from analysis_doc import write_doc
        write_doc()
    except Exception as exc:
        _log(f"doc refresh failed: {exc}")


def _grid():
    pending = []
    for spec in CONFIGS:
        for algo in ALGOS:
            if not _summary_path(spec["name"], algo["key"]).exists():
                pending.append((spec["name"], algo["key"]))
    _log(f"Pending jobs: {len(pending)}  workers={WORKERS}")
    active = []
    retried = set()
    while pending or active:
        while pending and len(active) < WORKERS:
            config_name, algo_key = pending.pop(0)
            handle = open(OUT / f"{config_name}__{algo_key}.log", "w")
            proc = subprocess.Popen(
                [sys.executable, "-u", str(Path(__file__).resolve()), "job", config_name, algo_key],
                cwd=str(ROOT),
                stdout=handle,
                stderr=subprocess.STDOUT,
            )
            active.append((config_name, algo_key, proc, handle))
            _log(f"started {config_name} {algo_key} pid={proc.pid}")
        _write_progress(pending, active)
        time.sleep(15)
        still = []
        finished_configs = set()
        for item in active:
            config_name, algo_key, proc, handle = item
            code = proc.poll()
            if code is None:
                still.append(item)
                continue
            handle.close()
            if code != 0:
                key = (config_name, algo_key)
                if key not in retried:
                    retried.add(key)
                    pending.append(key)
                    _log(f"retry {config_name} {algo_key} exit={code}")
                else:
                    _log(f"FAILED {config_name} {algo_key} exit={code}")
            finished_configs.add(config_name)
        active = still
        refreshed = False
        for config_name in finished_configs:
            if plot_config(config_name):
                _log(f"plots {config_name}")
                refreshed = True
        if refreshed:
            plot_overview()
            _refresh_doc()
    plot_overview()
    _log("Grid finished")


def _check():
    base, _path = load_instance()
    print("near", len(near_dealer_ids(base)), near_dealer_ids(base))
    print("national", national_dealer_ids())
    print("extra", len(_extra_dealers()), sum(count for _name, count in EXTRA_SITES))
    for spec in CONFIGS:
        data, meta = build_instance(base, spec)
        # Old pairs keep their distance when the dealer was in the original file.
        if spec["dealers"] in ("all", "large"):
            assert data["R"][1][2] == base["R"][1][2]
            assert data["A"][1][2] == base["A"][1][2]
        greedy = build_initial_trips(data)
        cost, _aux = evaluate(Solution(data, clone_trips(greedy)))
        print(
            f"{spec['name']:24} dealers={meta['dealers']:3} trucks={meta['trucks']:3} "
            f"demand={meta['demand_mode']} forecast={meta['forecast']:6} "
            f"greedy={cost.total:,.0f} tours={len(greedy)}"
        )


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if len(sys.argv) >= 2 and sys.argv[1] == "check":
        _check()
        return
    if len(sys.argv) >= 2 and sys.argv[1] == "job":
        _run_job(sys.argv[2], sys.argv[3])
        return
    if len(sys.argv) >= 2 and sys.argv[1] == "plots":
        for spec in CONFIGS:
            plot_config(spec["name"])
        plot_overview()
        return
    if len(sys.argv) >= 2 and sys.argv[1] == "doc":
        from analysis_doc import write_doc
        write_doc()
        return
    ensure_weights()
    _refresh_doc()
    _grid()
    from analysis_doc import write_doc
    write_doc()


if __name__ == "__main__":
    main()
