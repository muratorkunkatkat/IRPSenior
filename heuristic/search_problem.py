"""Shared solution, cost, and neighborhood for the large-instance metaheuristics.

Both searches start from the same dynamic-greedy plan and score it with the same
function, so a difference in the plots is a difference in the search, not in the
accounting.

Objective = transport + daily holding + daily backorder
            + leftover stock after the last day + dealer overflow penalty.

Transport uses R. Holding uses H on positive end-of-day stock. Backorder uses B
on the shortage carried that day. Leftover stock after the horizon is charged at
G per car. Stock above a dealer's storage limit is charged at G per car per day.
"""

from collections import defaultdict

# Same order of magnitude as the unsold penalty, so overflow is not free.
CAPACITY_PENALTY = 15000


class Trip:
    _next_uid = 1

    def __init__(self, day, truck, car_type, stops, qty):
        self.day = int(day)
        self.truck = int(truck)
        self.car_type = int(car_type)
        self.stops = [int(s) for s in stops]
        self.qty = [int(q) for q in qty]
        self.uid = Trip._next_uid
        Trip._next_uid += 1

    def copy(self):
        return Trip(self.day, self.truck, self.car_type, self.stops, self.qty)


class Cost:
    __slots__ = (
        "transport", "holding", "backorder", "unsold", "capacity",
        "backorder_units", "unsold_units", "holding_units", "delivered",
    )

    def __init__(self, transport, holding, backorder, unsold, capacity,
                 backorder_units, unsold_units, holding_units, delivered):
        self.transport = float(transport)
        self.holding = float(holding)
        self.backorder = float(backorder)
        self.unsold = float(unsold)
        self.capacity = float(capacity)
        self.backorder_units = float(backorder_units)
        self.unsold_units = float(unsold_units)
        self.holding_units = float(holding_units)
        self.delivered = int(delivered)

    @property
    def total(self):
        return self.transport + self.holding + self.backorder + self.unsold + self.capacity

    def as_dict(self):
        return {
            "transport": self.transport,
            "holding": self.holding,
            "backorder": self.backorder,
            "unsold": self.unsold,
            "capacity": self.capacity,
            "total": self.total,
        }


class Aux:
    __slots__ = ("by_day_type", "excess")

    def __init__(self, by_day_type, excess):
        self.by_day_type = by_day_type
        self.excess = excess


class Proposal:
    __slots__ = ("kind", "key", "undo", "redo")

    def __init__(self, kind, key, undo, redo):
        self.kind = kind
        self.key = key
        self.undo = undo
        self.redo = redo


class Solution:
    def __init__(self, data, trips):
        if not isinstance(data["A"], list):
            raise TypeError(
                "These searches read the list-format instance in data/data_exp.json."
            )
        self.data = data
        self.A = data["A"]
        self.R = data["R"]
        self.Q = data["Q"]
        self.F = data["F"]
        self.H = data["H"]
        self.B = data["B"]
        self.G = data["G"]
        self.V = data["V"]
        self.C = list(data["C"])
        self.dealers = list(data["nodes_no_factory"])
        self.n_days = len(data["T"])
        self.n_nodes = len(data["N"])
        self.n_types = len(self.C)
        self.days = list(data["T"])
        self.trips = list(trips)
        self.by_truck = defaultdict(list)
        for trip in self.trips:
            self.by_truck[trip.truck].append(trip)
        span = self.n_days * self.n_nodes * self.n_types
        self._delivered = [0] * span
        self._net = [[0] * self.n_types for _ in range(self.n_nodes)]
        self.cost = None
        self.aux = None

    def slot(self, day, node, car_type):
        return (day * self.n_nodes + node) * self.n_types + car_type


def clone_trips(trips):
    return [trip.copy() for trip in trips]


def build_initial_trips(data):
    """Dynamic greedy, with the cars dropped at each stop recorded.

    Same decisions as DynamicGreedyHeuristic: each free truck takes the car
    type with the largest residual demand, then visits nearest short dealers
    and unloads min(space left, cars short). Nothing is dumped at the last stop.
    """
    dealers = data["nodes_no_factory"]
    types = data["C"]
    trucks = data["K"]
    days = data["T"]
    A = data["A"]
    F = data["F"]
    Q = data["Q"]
    inventory = {i: {c: 0 for c in types} for i in dealers}
    locked_until = {k: 0 for k in trucks}
    trips = []

    for day in days:
        for truck in trucks:
            if locked_until[truck] > day:
                continue
            best_c = None
            max_unmet = -1
            for car in types:
                unmet = 0
                for dealer in dealers:
                    gap = F[dealer][car][day] - inventory[dealer][car]
                    if gap > 0:
                        unmet += gap
                if unmet > max_unmet and unmet > 0:
                    max_unmet = unmet
                    best_c = car
            if best_c is None:
                continue

            current = 0
            remaining = Q[best_c]
            route_time = 0
            stops = []
            qtys = []
            visited = {0}
            while remaining > 0:
                nearest = None
                nearest_time = None
                for dealer in dealers:
                    if dealer in visited:
                        continue
                    if inventory[dealer][best_c] >= F[dealer][best_c][day]:
                        continue
                    travel = A[current][dealer]
                    if nearest_time is None or travel < nearest_time:
                        nearest = dealer
                        nearest_time = travel
                if nearest is None:
                    break
                route_time += nearest_time
                need = F[nearest][best_c][day] - inventory[nearest][best_c]
                dropped = need if need < remaining else remaining
                inventory[nearest][best_c] += dropped
                remaining -= dropped
                current = nearest
                visited.add(nearest)
                stops.append(nearest)
                qtys.append(int(dropped))

            if stops:
                route_time += A[current][0]
                locked_until[truck] = day + route_time
                trips.append(Trip(day, truck, best_c, stops, qtys))

        for car in types:
            for dealer in dealers:
                inventory[dealer][car] -= F[dealer][car][day]

    return trips


def travel_time(A, trip):
    if not trip.stops or sum(trip.qty) <= 0:
        return 0
    prev = 0
    total = 0
    for stop in trip.stops:
        total += A[prev][stop]
        prev = stop
    total += A[prev][0]
    return total


def route_transport(R, trip):
    if not trip.stops or sum(trip.qty) <= 0:
        return 0.0
    prev = 0
    total = 0.0
    for stop in trip.stops:
        total += R[prev][stop]
        prev = stop
    total += R[prev][0]
    return total


def schedule_feasible(sol, truck):
    intervals = []
    for trip in sol.by_truck[truck]:
        if not trip.stops or sum(trip.qty) <= 0:
            continue
        if trip.day < 0 or trip.day >= sol.n_days:
            return False
        duration = travel_time(sol.A, trip)
        start = trip.day
        end = trip.day + max(duration, 1)
        intervals.append((start, end))
    intervals.sort()
    for earlier, later in zip(intervals, intervals[1:]):
        if earlier[1] > later[0]:
            return False
    return True


def trip_structurally_ok(sol, trip):
    if sum(trip.qty) > sol.Q[trip.car_type]:
        return False
    if any(q < 0 for q in trip.qty):
        return False
    if len(trip.stops) != len(trip.qty):
        return False
    if len(set(trip.stops)) != len(trip.stops):
        return False
    if any(stop == 0 or stop >= sol.n_nodes for stop in trip.stops):
        return False
    return True


def affected_ok(sol, trips):
    trucks = set()
    for trip in trips:
        if not trip_structurally_ok(sol, trip):
            return False
        trucks.add(trip.truck)
    return all(schedule_feasible(sol, truck) for truck in trucks)


def snapshot(trips):
    return [(trip, trip.day, trip.car_type, trip.stops[:], trip.qty[:]) for trip in trips]


def restore(saved):
    for trip, day, car_type, stops, qty in saved:
        trip.day = day
        trip.car_type = car_type
        trip.stops = stops[:]
        trip.qty = qty[:]


def _changed(saved):
    for trip, day, car_type, stops, qty in saved:
        if (trip.day != day or trip.car_type != car_type
                or trip.stops != stops or trip.qty != qty):
            return True
    return False


def finish(sol, affected, saved, kind, key):
    if not _changed(saved) or not affected_ok(sol, affected):
        restore(saved)
        return None
    after = snapshot(affected)

    def undo():
        restore(saved)

    def redo():
        restore(after)

    return Proposal(kind, key, undo, redo)


def evaluate(sol):
    delivered = sol._delivered
    for index in range(len(delivered)):
        delivered[index] = 0

    transport = 0.0
    shipped = 0
    for trip in sol.trips:
        if not trip.stops:
            continue
        transport += route_transport(sol.R, trip)
        day = trip.day
        car = trip.car_type
        for stop, qty in zip(trip.stops, trip.qty):
            if qty <= 0:
                continue
            delivered[sol.slot(day, stop, car)] += qty
            shipped += qty

    nets = sol._net
    for dealer in sol.dealers:
        row = nets[dealer]
        for car in sol.C:
            row[car] = 0

    holding = 0.0
    backorder = 0.0
    holding_units = 0.0
    backorder_units = 0.0
    capacity = 0.0
    by_day_type = {}
    F = sol.F
    H = sol.H
    B = sol.B
    V = sol.V
    n_types = sol.n_types

    for day in sol.days:
        day_base = day * sol.n_nodes * n_types
        for dealer in sol.dealers:
            row = nets[dealer]
            on_hand = 0
            dealer_base = day_base + dealer * n_types
            for car in sol.C:
                net = row[car] + delivered[dealer_base + car] - F[dealer][car][day]
                row[car] = net
                if net >= 0:
                    holding += H[car] * net
                    holding_units += net
                    on_hand += net
                else:
                    short = -net
                    backorder += B[car] * short
                    backorder_units += short
                    bucket = by_day_type.get((day, car))
                    if bucket is None:
                        bucket = []
                        by_day_type[(day, car)] = bucket
                    bucket.append((dealer, short))
            overflow = on_hand - V[dealer]
            if overflow > 0:
                capacity += CAPACITY_PENALTY * overflow

    unsold = 0.0
    unsold_units = 0.0
    excess = {}
    G = sol.G
    for dealer in sol.dealers:
        row = nets[dealer]
        for car in sol.C:
            net = row[car]
            if net > 0:
                unsold += G[car] * net
                unsold_units += net
                excess[(dealer, car)] = net

    cost = Cost(
        transport, holding, backorder, unsold, capacity,
        backorder_units, unsold_units, holding_units, shipped,
    )
    return cost, Aux(by_day_type, excess)


def _insertion_pos(A, stops, dealer, rng):
    best = []
    for pos in range(len(stops) + 1):
        before = 0 if pos == 0 else stops[pos - 1]
        after = 0 if pos == len(stops) else stops[pos]
        extra = A[before][dealer] + A[dealer][after] - A[before][after]
        best.append((extra, pos))
    best.sort()
    shortlist = best[:min(2, len(best))]
    return rng.choice(shortlist)[1]


def _active(sol):
    return [trip for trip in sol.trips if trip.stops and sum(trip.qty) > 0]


def try_swap(sol, rng):
    pool = [trip for trip in sol.trips if len(trip.stops) >= 2]
    if not pool:
        return None
    trip = rng.choice(pool)
    i, j = rng.sample(range(len(trip.stops)), 2)
    saved = snapshot([trip])
    trip.stops[i], trip.stops[j] = trip.stops[j], trip.stops[i]
    trip.qty[i], trip.qty[j] = trip.qty[j], trip.qty[i]
    a, b = sorted((trip.stops[i], trip.stops[j]))
    return finish(sol, [trip], saved, "swap", ("swap", trip.uid, a, b))


def try_relocate_in(sol, rng):
    pool = [trip for trip in sol.trips if len(trip.stops) >= 2]
    if not pool:
        return None
    trip = rng.choice(pool)
    saved = snapshot([trip])
    index = rng.randrange(len(trip.stops))
    dealer = trip.stops.pop(index)
    qty = trip.qty.pop(index)
    new_index = rng.randrange(len(trip.stops) + 1)
    trip.stops.insert(new_index, dealer)
    trip.qty.insert(new_index, qty)
    return finish(sol, [trip], saved, "relocate-in", ("in", trip.uid, dealer))


def try_reverse(sol, rng):
    pool = [trip for trip in sol.trips if len(trip.stops) >= 2]
    if not pool:
        return None
    trip = rng.choice(pool)
    i, j = sorted(rng.sample(range(len(trip.stops)), 2))
    saved = snapshot([trip])
    trip.stops[i:j + 1] = trip.stops[i:j + 1][::-1]
    trip.qty[i:j + 1] = trip.qty[i:j + 1][::-1]
    return finish(sol, [trip], saved, "reverse", ("reverse", trip.uid, i, j))


def try_relocate_out(sol, rng):
    donors = _active(sol)
    if len(donors) < 2:
        return None
    src = rng.choice(donors)
    same_type = [trip for trip in donors if trip is not src and trip.car_type == src.car_type]
    if not same_type:
        return None
    dst = rng.choice(same_type)
    index = rng.randrange(len(src.stops))
    dealer = src.stops[index]
    qty = src.qty[index]
    if sum(dst.qty) + qty > sol.Q[dst.car_type]:
        return None
    saved = snapshot([src, dst])
    src.stops.pop(index)
    src.qty.pop(index)
    if dealer in dst.stops:
        dst.qty[dst.stops.index(dealer)] += qty
    else:
        pos = _insertion_pos(sol.A, dst.stops, dealer, rng)
        dst.stops.insert(pos, dealer)
        dst.qty.insert(pos, qty)
    return finish(sol, [src, dst], saved, "relocate-out",
                  ("out", src.uid, dst.uid, dealer))


def try_add(sol, rng):
    if sol.aux is None:
        return None
    options = []
    for trip in sol.trips:
        spare = sol.Q[trip.car_type] - sum(trip.qty)
        if spare <= 0:
            continue
        needy = sol.aux.by_day_type.get((trip.day, trip.car_type))
        if needy:
            options.append((trip, spare, needy))
    if not options:
        return None
    trip, spare, needy = rng.choice(options)
    dealer, short = rng.choice(needy)
    add = min(spare, int(short))
    if add <= 0:
        return None
    add = rng.randint(1, add)
    saved = snapshot([trip])
    if dealer in trip.stops:
        trip.qty[trip.stops.index(dealer)] += add
    else:
        pos = _insertion_pos(sol.A, trip.stops, dealer, rng)
        trip.stops.insert(pos, dealer)
        trip.qty.insert(pos, add)
    return finish(sol, [trip], saved, "add", ("add", trip.uid, dealer, trip.day))


def try_trim(sol, rng):
    if sol.aux is None or not sol.aux.excess:
        return None
    dealer, car = rng.choice(list(sol.aux.excess.keys()))
    limit = int(sol.aux.excess[(dealer, car)])
    if limit <= 0:
        return None
    serving = [trip for trip in sol.trips
               if trip.car_type == car and dealer in trip.stops and sum(trip.qty) > 0]
    if not serving:
        return None
    serving.sort(key=lambda trip: trip.day)
    trip = serving[-1]
    index = trip.stops.index(dealer)
    cut = min(trip.qty[index], limit)
    if cut <= 0:
        return None
    cut = rng.randint(1, cut)
    saved = snapshot([trip])
    trip.qty[index] -= cut
    if trip.qty[index] == 0:
        trip.stops.pop(index)
        trip.qty.pop(index)
    return finish(sol, [trip], saved, "trim", ("trim", trip.uid, dealer))


def _fill_route(sol, day, car_type, rng):
    needy = [(dealer, int(short))
             for dealer, short in sol.aux.by_day_type.get((day, car_type), [])
             if short > 0]
    remaining = sol.Q[car_type]
    current = 0
    stops = []
    qtys = []
    A = sol.A
    while remaining > 0 and needy:
        needy.sort(key=lambda item: A[current][item[0]])
        dealer, short = rng.choice(needy[:min(3, len(needy))])
        take = min(remaining, short)
        if take <= 0:
            needy = [item for item in needy if item[0] != dealer]
            continue
        stops.append(dealer)
        qtys.append(take)
        remaining -= take
        current = dealer
        needy = [item for item in needy if item[0] != dealer]
    return stops, qtys


def try_rebuild(sol, rng):
    if sol.aux is None:
        return None
    pool = _active(sol)
    if not pool:
        return None
    trip = rng.choice(pool)
    totals = []
    for car in sol.C:
        total = sum(short for _, short in sol.aux.by_day_type.get((trip.day, car), []))
        totals.append((total, car))
    totals.sort(reverse=True)
    if rng.random() < 0.8 and totals[0][0] > 0:
        new_type = totals[0][1]
    else:
        new_type = rng.choice(sol.C)
    stops, qtys = _fill_route(sol, trip.day, new_type, rng)
    if not stops:
        return None
    saved = snapshot([trip])
    trip.car_type = new_type
    trip.stops = stops
    trip.qty = qtys
    return finish(sol, [trip], saved, "rebuild", ("rebuild", trip.uid, new_type, trip.day))


def try_shift(sol, rng):
    pool = _active(sol)
    if not pool:
        return None
    trip = rng.choice(pool)
    for _ in range(6):
        new_day = trip.day + rng.choice((-3, -2, -1, 1, 2, 3))
        if new_day < 0 or new_day >= sol.n_days or new_day == trip.day:
            continue
        saved = snapshot([trip])
        trip.day = new_day
        proposal = finish(sol, [trip], saved, "shift", ("shift", trip.uid, new_day))
        if proposal is not None:
            return proposal
    return None


_BUILDERS = (
    (try_swap, 1.0),
    (try_relocate_in, 1.0),
    (try_reverse, 1.0),
    (try_relocate_out, 1.5),
    (try_add, 2.0),
    (try_trim, 1.2),
    (try_rebuild, 1.6),
    (try_shift, 0.8),
)
_MOVE_FUNCS = [item[0] for item in _BUILDERS]
_MOVE_WEIGHTS = [item[1] for item in _BUILDERS]


def propose_move(sol, rng):
    for _ in range(40):
        builder = rng.choices(_MOVE_FUNCS, weights=_MOVE_WEIGHTS, k=1)[0]
        proposal = builder(sol, rng)
        if proposal is not None:
            return proposal
    return None


def count_active(trips):
    return sum(1 for trip in trips if trip.stops and sum(trip.qty) > 0)
