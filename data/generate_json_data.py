"""Company-scale instance for the 71-dealer Fiat network.

Demand, truck capacity, and fleet size follow the operating averages supplied
for IE 402. Geography is a Bursa factory plus 71 dealer sites spread across
Türkiye in proportion to where Fiat dealers actually sit (large cities have
several sites). Seed 44 keeps the file reproducible.

Car types:
    0 Egea Cross, 1 Egea Sedan, 2 Scudo, 3 Scudo kamyonet, 4 Ulysse
"""

import json
import math
import os
import random

# Average cars demanded per dealer per day, by car type.
DAILY_MEAN = [0.6358, 0.9972, 0.2962, 0.0610, 0.0524]
# Average dispatches per day, by car type. These are tours, not distinct trucks.
# 8+12+6+1+1 = 28. The month has 715 dispatches in total (28 * 25.5 ≈ 715).
DAILY_DISPATCHES = [8, 12, 6, 1, 1]
MONTHLY_DISPATCHES = 715

# Deck space. Egea models ride a standard car carrier; the three larger
# vehicles take a slot and a half, so the same truck holds fewer of them.
# Each capacity is the smallest standard deck that still sits above the
# average load implied by daily demand / daily dispatches.
TRUCK_CAPACITY = [8, 8, 6, 6, 6]

DAYS = 30
# Road distance ≈ great-circle × this factor (Turkish highways are not straight).
ROAD_FACTOR = 1.35
# A leg shorter than this is chained inside the same working day (time 0).
# The dispatch still occupies that calendar day. Longer legs take whole days
# at this effective speed, including breaks and unloading.
SAME_DAY_KM = 280
KM_PER_DAY = 500
COST_PER_KM = 1.5

# (city, latitude, longitude, how many dealer sites, yard capacity)
# Counts sum to 71. Istanbul through Kocaeli match the dense western network.
CITIES = [
    ("İstanbul", 41.015, 28.979, 11, 80),
    ("Ankara", 39.933, 32.859, 5, 55),
    ("İzmir", 38.424, 27.143, 4, 55),
    ("Bursa", 40.195, 29.061, 3, 55),
    ("Antalya", 36.897, 30.713, 3, 50),
    ("Adana", 37.000, 35.321, 2, 45),
    ("Konya", 37.874, 32.493, 2, 45),
    ("Gaziantep", 37.066, 37.378, 2, 45),
    ("Kocaeli", 40.765, 29.940, 2, 45),
    ("Mersin", 36.812, 34.641, 2, 45),
    ("Diyarbakır", 37.915, 40.230, 1, 36),
    ("Şanlıurfa", 37.159, 38.797, 1, 36),
    ("Hatay", 36.202, 36.160, 1, 36),
    ("Manisa", 38.619, 27.428, 1, 36),
    ("Kayseri", 38.721, 35.487, 1, 36),
    ("Samsun", 41.286, 36.330, 1, 36),
    ("Balıkesir", 39.648, 27.883, 1, 36),
    ("Kahramanmaraş", 37.575, 36.923, 1, 36),
    ("Van", 38.501, 43.373, 1, 36),
    ("Aydın", 37.856, 27.841, 1, 36),
    ("Denizli", 37.776, 29.086, 1, 36),
    ("Tekirdağ", 40.978, 27.511, 1, 36),
    ("Sakarya", 40.694, 30.435, 1, 36),
    ("Muğla", 37.215, 28.364, 1, 36),
    ("Eskişehir", 39.777, 30.521, 1, 36),
    ("Trabzon", 41.002, 39.717, 1, 36),
    ("Malatya", 38.355, 38.309, 1, 36),
    ("Erzurum", 39.905, 41.265, 1, 36),
    ("Ordu", 40.984, 37.878, 1, 36),
    ("Afyonkarahisar", 38.751, 30.556, 1, 36),
    ("Sivas", 39.748, 37.016, 1, 36),
    ("Zonguldak", 41.456, 31.799, 1, 36),
    ("Elazığ", 38.681, 39.226, 1, 36),
    ("Ağrı", 39.719, 43.051, 1, 36),
    ("Batman", 37.881, 41.129, 1, 36),
    ("Kütahya", 39.417, 29.983, 1, 36),
    ("Edirne", 41.677, 26.556, 1, 36),
    ("Çanakkale", 40.155, 26.414, 1, 36),
    ("Uşak", 38.682, 29.408, 1, 36),
    ("Isparta", 37.764, 30.556, 1, 36),
    ("Bolu", 40.736, 31.606, 1, 36),
    ("Rize", 41.020, 40.523, 1, 36),
    ("Aksaray", 38.368, 34.037, 1, 36),
    ("Kırklareli", 41.733, 27.225, 1, 36),
    ("Yalova", 40.650, 29.267, 1, 36),
]

FACTORY = (40.2262, 28.9877)  # Tofaş plant, Bursa


def haversine_km(lat1, lon1, lat2, lon2):
    radius = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlon / 2) ** 2
    return 2 * radius * math.asin(math.sqrt(a))


def travel_days(road_km):
    """Integer days a loaded carrier needs for one leg."""
    if road_km < 1 or road_km <= SAME_DAY_KM:
        return 0
    return max(1, int(math.ceil(road_km / KM_PER_DAY)))


def spread_sites(lat, lon, count):
    if count == 1:
        return [(lat, lon)]
    sites = []
    for index in range(count):
        angle = 2 * math.pi * index / count
        # About 8–12 km between neighbouring sites in the same city.
        sites.append((lat + 0.08 * math.cos(angle), lon + 0.10 * math.sin(angle)))
    return sites


def probabilities_012(mean):
    """Poisson shape on {0, 1, 2}, shifted so the mean is exact."""
    base = math.exp(-mean)
    raw = [base, base * mean, base * mean * mean / 2.0]
    scale = sum(raw)
    probs = [value / scale for value in raw]
    gap = mean - (probs[1] + 2 * probs[2])
    if gap > 0:
        move = min(probs[0], gap)
        probs[0] -= move
        probs[1] += move
        gap -= move
        move = min(probs[0], gap / 2.0)
        probs[0] -= move
        probs[2] += move
        gap -= 2 * move
        move = min(probs[1], max(gap, 0.0))
        probs[1] -= move
        probs[2] += move
    elif gap < 0:
        need = -gap
        move = min(probs[2], need / 2.0)
        probs[2] -= move
        probs[0] += move
        need -= 2 * move
        move = min(probs[2], need)
        probs[2] -= move
        probs[1] += move
        need -= move
        move = min(probs[1], need)
        probs[1] -= move
        probs[0] += move
    return probs


def demand_pool(mean, n_obs, rng):
    """Exactly n_obs draws from {0, 1, 2} whose average equals round(mean * n) / n."""
    total = int(round(mean * n_obs))
    probs = probabilities_012(mean)
    low = max(0, total - n_obs)
    high = total // 2
    n_twos = int(round(probs[2] * n_obs))
    n_twos = min(max(n_twos, low), high)
    n_ones = total - 2 * n_twos
    n_zeros = n_obs - n_ones - n_twos
    pool = [0] * n_zeros + [1] * n_ones + [2] * n_twos
    rng.shuffle(pool)
    return pool, probs


def dealer_sites():
    sites = []
    for name, lat, lon, count, yard in CITIES:
        for site_lat, site_lon in spread_sites(lat, lon, count):
            sites.append((name, site_lat, site_lon, yard))
    if len(sites) != 71:
        raise RuntimeError(f"expected 71 dealers, built {len(sites)}")
    return sites


def estimate_route_days(travel, demand, capacity, dealers, days, types):
    """Mean length of a full nearest-neighbour tour, one month, unlimited fleet."""
    durations = []
    for day in days:
        for car in types:
            remaining = {
                dealer: demand[dealer][car][day]
                for dealer in dealers
                if demand[dealer][car][day] > 0
            }
            while remaining:
                cap = capacity[car]
                current = 0
                route_time = 0
                while cap > 0 and remaining:
                    nxt = min(remaining, key=lambda dealer: travel[current][dealer])
                    route_time += travel[current][nxt]
                    take = min(cap, remaining[nxt])
                    cap -= take
                    if take >= remaining[nxt]:
                        del remaining[nxt]
                    else:
                        remaining[nxt] -= take
                    current = nxt
                route_time += travel[current][0]
                # A same-day tour still uses the truck on the dispatch day.
                durations.append(max(route_time, 1))
    if not durations:
        return 1.0
    return sum(durations) / len(durations)


def generate_json():
    rng = random.Random(44)
    sites = dealer_sites()
    n_nodes = 1 + len(sites)
    days = list(range(DAYS))
    types = list(range(len(DAILY_MEAN)))

    coords = [FACTORY] + [(lat, lon) for _name, lat, lon, _yard in sites]
    names = ["Bursa factory"] + [name for name, _lat, _lon, _yard in sites]

    road = [[0.0] * n_nodes for _ in range(n_nodes)]
    travel = [[0] * n_nodes for _ in range(n_nodes)]
    cost = [[0.0] * n_nodes for _ in range(n_nodes)]
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            km = haversine_km(coords[i][0], coords[i][1], coords[j][0], coords[j][1])
            km *= ROAD_FACTOR
            road[i][j] = road[j][i] = km
            leg = travel_days(km)
            travel[i][j] = travel[j][i] = leg
            fare = round(km * COST_PER_KM, 2)
            cost[i][j] = cost[j][i] = fare

    dealers = list(range(1, n_nodes))
    n_obs = len(dealers) * DAYS
    demand = []
    realized = []
    weights = []
    for i in range(n_nodes):
        demand.append([])
        for car in types:
            if i == 0:
                demand[i].append([0] * DAYS)
            else:
                demand[i].append([0] * DAYS)

    for car, mean in enumerate(DAILY_MEAN):
        pool, probs = demand_pool(mean, n_obs, rng)
        weights.append(probs)
        cursor = 0
        total = 0
        for dealer in dealers:
            for day in days:
                value = pool[cursor]
                cursor += 1
                demand[dealer][car][day] = value
                total += value
        realized.append(total / n_obs)

    yards = [0] + [yard for _name, _lat, _lon, yard in sites]
    avg_route = estimate_route_days(travel, demand, TRUCK_CAPACITY, dealers, days, types)
    # Fixed operating fleet. The 715-dispatch formula gave ~98; the team set 75.
    n_trucks = 75
    formula_fleet = max(1, int(round(MONTHLY_DISPATCHES * avg_route / DAYS)))

    data = {
        "C": types,
        "K": list(range(n_trucks)),
        "N": list(range(n_nodes)),
        "T": days,
        "nodes_no_factory": dealers,
        "A": travel,
        "R": cost,
        "Q": list(TRUCK_CAPACITY),
        "V": yards,
        "B": [500] * len(types),
        "H": [20] * len(types),
        "G": [15000] * len(types),
        "P": [1 if day == 0 else 0 for day in days],
        "F": demand,
    }

    os.makedirs("data", exist_ok=True)
    with open("data/data_exp.json", "w") as handle:
        json.dump(data, handle)

    labels = ["Egea Cross", "Egea Sedan", "Scudo", "Scudo kamyonet", "Ulysse"]
    print(f"nodes {n_nodes}  dealers {len(dealers)}  trucks {n_trucks}  days {DAYS}")
    print(
        f"average route {avg_route:.2f} days  ->  fleet {n_trucks} trucks "
        f"(formula would give {formula_fleet} for {MONTHLY_DISPATCHES} monthly dispatches)"
    )
    print(f"{'type':<18} {'target':>8} {'realized':>10} {'P0':>7} {'P1':>7} {'P2':>7} {'Q':>4}")
    for car, label in enumerate(labels):
        p0, p1, p2 = weights[car]
        print(f"{label:<18} {DAILY_MEAN[car]:8.4f} {realized[car]:10.4f} {p0:7.3f} {p1:7.3f} {p2:7.3f} {TRUCK_CAPACITY[car]:4d}")

    samples = {
        "İstanbul": None,
        "Ankara": None,
        "İzmir": None,
        "Van": None,
        "Trabzon": None,
        "Antalya": None,
    }
    for index, name in enumerate(names):
        if name in samples and samples[name] is None:
            samples[name] = index
    print("one-way from Bursa factory:")
    for name, index in samples.items():
        print(f"  {name:<12} {road[0][index]:7.0f} km   {travel[0][index]} day(s)   cost {cost[0][index]:,.0f}")


if __name__ == "__main__":
    generate_json()
