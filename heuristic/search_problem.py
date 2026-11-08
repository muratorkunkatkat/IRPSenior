"""Shared solution, cost, and neighborhood for the large-instance metaheuristics.

Both searches start from the same dynamic-greedy plan and score it with the same
function, so a difference in the plots is a difference in the search, not in the
accounting.

Objective = transport + daily holding + daily backorder
            + leftover stock after the last day + dealer overflow penalty.

Transport uses R. Holding uses H on positive end-of-day stock. Backorder uses B
on the shortage carried that day. Leftover stock after the horizon is charged at
G per car. Stock above a dealer's storage limit is charged at G per car per day.

Seven moves, drawn with equal weight unless the caller changes the weights.
Driving-cost choices use kilometers (R). A move that cannot be built is skipped.
After one is built, it is thrown away if two trips of the same truck overlap in
time. The full objective is scored afterwards, so a move is still offered when
holding, backorder, or unsold goes up.

  rebuild        reorder one random tour into its cheapest driving order
  hand-over      move one random dealer onto one other same-type tour, cheapest slot
  exchange       swap two random dealers between two same-type tours, cheapest slots
  add-car        drop one extra car at a random dealer, if the deck has room
  remove-car     drop one fewer car at a random dealer; a zero drop stays on the tour
  add-dealer     insert a missing dealer at the cheapest slot, quantity 0.
                 The target may be an existing tour or a truck that is free that day.
  remove-dealer  delete one random dealer from a tour. The tour is removed
                 when that dealer was its last stop.
"""

from collections import defaultdict
from itertools import permutations

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
    __slots__ = ("by_day_type", "excess", "spare", "short")

    def __init__(self, by_day_type, excess, spare, short):
        self.by_day_type = by_day_type
        self.excess = excess
        self.spare = spare
        self.short = short


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
        self.trucks = list(data["K"])
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


def build_random_trips(data, rng, attempts=8000):
    """A nonsense plan. Day, truck, model, dealers, and drop sizes are all draws.

    Demand, distance, and stock are ignored. A draw is kept only when the
    neighborhood can still edit it: the dealers on a tour are distinct, the
    dropped cars fit on the deck, and that truck is free for the travel window.
    """
    dealers = list(data["nodes_no_factory"])
    types = list(data["C"])
    trucks = list(data["K"])
    days = list(data["T"])
    A = data["A"]
    Q = data["Q"]
    busy = {truck: [] for truck in trucks}
    trips = []
    if not dealers or not types or not trucks or not days:
        return trips

    for _ in range(attempts):
        car = int(rng.choice(types))
        cap = int(Q[car])
        if cap < 1:
            continue
        n_stops = rng.randint(1, min(8, len(dealers)))
        stops = rng.sample(dealers, n_stops)
        qty = [rng.randint(1, cap) for _stop in stops]
        if sum(qty) > cap:
            continue
        day = int(rng.choice(days))
        truck = int(rng.choice(trucks))
        duration = 0
        prev = 0
        for stop in stops:
            duration += A[prev][stop]
            prev = stop
        duration += A[prev][0]
        start = day
        end = start + max(duration, 1)
        if any(other_start < end and start < other_end for other_start, other_end in busy[truck]):
            continue
        busy[truck].append((start, end))
        trips.append(Trip(day, truck, car, stops, qty))
    return trips


def travel_time(A, trip):
    if not trip.stops:
        return 0
    prev = 0
    total = 0
    for stop in trip.stops:
        total += A[prev][stop]
        prev = stop
    total += A[prev][0]
    return total


def route_transport(R, trip):
    if not trip.stops:
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
    if not trip.stops:
        return True
    if len(trip.stops) != len(trip.qty):
        return False
    if any(q < 0 for q in trip.qty):
        return False
    if len(set(trip.stops)) != len(trip.stops):
        return False
    if any(stop == 0 or stop >= sol.n_nodes for stop in trip.stops):
        return False
    cap = sol.Q[trip.car_type]
    factory_load = sum(trip.qty)
    if factory_load < 0 or factory_load > cap:
        return False
    load = factory_load
    for qty in trip.qty:
        load -= qty
        if load < -1e-9 or load > cap + 1e-9:
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
            if qty == 0:
                continue
            delivered[sol.slot(day, stop, car)] += qty
            if qty > 0:
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
    spare = {}
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
                opening = row[car]
                if opening > 0:
                    bucket = spare.get((day, car))
                    if bucket is None:
                        bucket = []
                        spare[(day, car)] = bucket
                    bucket.append((dealer, opening))
                net = opening + delivered[dealer_base + car] - F[dealer][car][day]
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
    return cost, Aux(by_day_type, excess, spare, by_day_type)


def _active(sol):
    return [trip for trip in sol.trips if trip.stops and sum(q for q in trip.qty if q > 0) > 0]


def _path_cost(R, stops):
    if not stops:
        return 0.0
    prev = 0
    total = 0.0
    for stop in stops:
        total += R[prev][stop]
        prev = stop
    total += R[prev][0]
    return total


def _insertion_extra(R, stops, dealer):
    best_pos = 0
    best_extra = None
    for pos in range(len(stops) + 1):
        before = 0 if pos == 0 else stops[pos - 1]
        after = 0 if pos == len(stops) else stops[pos]
        extra = R[before][dealer] + R[dealer][after] - R[before][after]
        if best_extra is None or extra < best_extra:
            best_extra = extra
            best_pos = pos
    return best_pos, 0.0 if best_extra is None else best_extra


def _removal_saving(R, stops, index):
    dealer = stops[index]
    before = 0 if index == 0 else stops[index - 1]
    after = 0 if index == len(stops) - 1 else stops[index + 1]
    return R[before][dealer] + R[dealer][after] - R[before][after]


def _apply_order(trip, order, qty_of):
    trip.stops = list(order)
    trip.qty = [qty_of[stop] for stop in order]


def _same_day_groups(sol):
    groups = defaultdict(list)
    for trip in _active(sol):
        groups[(trip.day, trip.car_type)].append(trip)
    return groups


def _qty_map(trip):
    return {stop: qty for stop, qty in zip(trip.stops, trip.qty)}


def _interval(sol, trip):
    duration = travel_time(sol.A, trip)
    start = trip.day
    return start, start + max(duration, 1)


def _truck_free(sol, truck, start, end, ignore=None):
    for trip in sol.by_truck[truck]:
        if trip is ignore:
            continue
        other_start, other_end = _interval(sol, trip)
        if other_start < end and start < other_end:
            return False
    return True


def _attach(sol, trip):
    sol.trips.append(trip)
    sol.by_truck[trip.truck].append(trip)


def _detach(sol, trip):
    sol.trips.remove(trip)
    sol.by_truck[trip.truck].remove(trip)


def _best_order(R, stops):
    """Cheapest factory-to-factory order of these dealers. Exact up to 11 stops."""
    n = len(stops)
    if n <= 1:
        return list(stops), _path_cost(R, stops)
    if n > 11:
        return list(stops), _path_cost(R, stops)
    if n <= 8:
        best_perm = None
        best_cost = None
        for perm in permutations(range(n)):
            cost = 0.0
            prev = 0
            too_dear = False
            for index in perm:
                node = stops[index]
                cost += R[prev][node]
                prev = node
                if best_cost is not None and cost >= best_cost:
                    too_dear = True
                    break
            if too_dear:
                continue
            cost += R[prev][0]
            if best_cost is None or cost < best_cost:
                best_cost = cost
                best_perm = perm
        return [stops[index] for index in best_perm], best_cost

    size = 1 << n
    width = n
    inf = float("inf")
    dp = [inf] * (size * width)
    parent = [-1] * (size * width)
    for end in range(n):
        dp[(1 << end) * width + end] = R[0][stops[end]]
    for mask in range(size):
        for end in range(n):
            if (mask & (1 << end)) == 0:
                continue
            base = dp[mask * width + end]
            if base == inf:
                continue
            for nxt in range(n):
                bit = 1 << nxt
                if mask & bit:
                    continue
                cand = base + R[stops[end]][stops[nxt]]
                slot = (mask | bit) * width + nxt
                if cand < dp[slot]:
                    dp[slot] = cand
                    parent[slot] = end
    full = size - 1
    best_cost = inf
    last = 0
    for end in range(n):
        cand = dp[full * width + end] + R[stops[end]][0]
        if cand < best_cost:
            best_cost = cand
            last = end
    order = []
    mask = full
    end = last
    while end >= 0:
        order.append(end)
        prev = parent[mask * width + end]
        mask ^= 1 << end
        end = prev
    order.reverse()
    return [stops[index] for index in order], best_cost


def _place_dealer(stops, qty, dealer, amount, R):
    if dealer in stops:
        qty[stops.index(dealer)] += amount
        return
    pos, _extra = _insertion_extra(R, stops, dealer)
    stops.insert(pos, dealer)
    qty.insert(pos, amount)


def try_rebuild(sol, rng):
    pool = [trip for trip in sol.trips if len(trip.stops) >= 2]
    if not pool:
        return None
    trip = rng.choice(pool)
    order, cost = _best_order(sol.R, trip.stops)
    if order == trip.stops or cost + 1e-9 >= _path_cost(sol.R, trip.stops):
        return None
    saved = snapshot([trip])
    _apply_order(trip, order, _qty_map(trip))
    return finish(sol, [trip], saved, "rebuild", ("rebuild", trip.uid, tuple(order)))


def try_hand_over(sol, rng):
    sources = [trip for trip in sol.trips if trip.stops]
    if not sources:
        return None
    src = rng.choice(sources)
    index = rng.randrange(len(src.stops))
    dealer = src.stops[index]
    qty = src.qty[index]
    receivers = []
    for trip in sol.trips:
        if trip is src or trip.car_type != src.car_type:
            continue
        if sum(trip.qty) + qty > sol.Q[src.car_type]:
            continue
        receivers.append(trip)
    if not receivers:
        return None
    dst = rng.choice(receivers)
    saved = snapshot([src, dst])
    src.stops.pop(index)
    src.qty.pop(index)
    _place_dealer(dst.stops, dst.qty, dealer, qty, sol.R)
    pair = (min(src.uid, dst.uid), max(src.uid, dst.uid))
    return finish(sol, [src, dst], saved, "hand-over", ("hand-over", dealer, pair))


def try_exchange(sol, rng):
    by_type = defaultdict(list)
    for trip in sol.trips:
        if trip.stops:
            by_type[trip.car_type].append(trip)
    types = [car for car, trips in by_type.items() if len(trips) >= 2]
    if not types:
        return None
    car = rng.choice(types)
    src, dst = rng.sample(by_type[car], 2)
    i = rng.randrange(len(src.stops))
    j = rng.randrange(len(dst.stops))
    dealer_a, qty_a = src.stops[i], src.qty[i]
    dealer_b, qty_b = dst.stops[j], dst.qty[j]
    if dealer_a == dealer_b or dealer_a in dst.stops or dealer_b in src.stops:
        return None
    cap = sol.Q[car]
    if sum(src.qty) - qty_a + qty_b > cap or sum(dst.qty) - qty_b + qty_a > cap:
        return None
    src_rest = src.stops[:i] + src.stops[i + 1:]
    dst_rest = dst.stops[:j] + dst.stops[j + 1:]
    pos_b, _extra_b = _insertion_extra(sol.R, src_rest, dealer_b)
    pos_a, _extra_a = _insertion_extra(sol.R, dst_rest, dealer_a)
    trial_src = src_rest[:pos_b] + [dealer_b] + src_rest[pos_b:]
    trial_dst = dst_rest[:pos_a] + [dealer_a] + dst_rest[pos_a:]
    qty_src = {stop: amount for stop, amount in zip(src.stops, src.qty) if stop != dealer_a}
    qty_dst = {stop: amount for stop, amount in zip(dst.stops, dst.qty) if stop != dealer_b}
    qty_src[dealer_b] = qty_b
    qty_dst[dealer_a] = qty_a
    saved = snapshot([src, dst])
    _apply_order(src, trial_src, qty_src)
    _apply_order(dst, trial_dst, qty_dst)
    dealers = tuple(sorted((dealer_a, dealer_b)))
    tours = (min(src.uid, dst.uid), max(src.uid, dst.uid))
    return finish(sol, [src, dst], saved, "exchange", ("exchange", dealers, tours))


def try_add_car(sol, rng):
    pool = [trip for trip in sol.trips if trip.stops]
    if not pool:
        return None
    trip = rng.choice(pool)
    if sum(trip.qty) >= sol.Q[trip.car_type]:
        return None
    index = rng.randrange(len(trip.stops))
    saved = snapshot([trip])
    trip.qty[index] += 1
    return finish(sol, [trip], saved, "add-car",
                  ("add-car", trip.uid, trip.stops[index], trip.day))


def try_remove_car(sol, rng):
    pool = [trip for trip in sol.trips if any(qty > 0 for qty in trip.qty)]
    if not pool:
        return None
    trip = rng.choice(pool)
    choices = [i for i, qty in enumerate(trip.qty) if qty > 0]
    index = rng.choice(choices)
    dealer = trip.stops[index]
    saved = snapshot([trip])
    trip.qty[index] -= 1
    return finish(sol, [trip], saved, "remove-car",
                  ("remove-car", trip.uid, dealer, trip.day, trip.qty[index]))


def _add_dealer_on_route(sol, rng):
    if not sol.trips:
        return None
    trip = rng.choice(sol.trips)
    missing = [dealer for dealer in sol.dealers if dealer not in trip.stops]
    if not missing:
        return None
    dealer = int(rng.choice(missing))
    pos, _extra = _insertion_extra(sol.R, trip.stops, dealer)
    saved = snapshot([trip])
    trip.stops.insert(pos, dealer)
    trip.qty.insert(pos, 0)
    return finish(sol, [trip], saved, "add-dealer", ("add-dealer", trip.uid, dealer, pos))


def _add_dealer_on_free_truck(sol, rng):
    day = int(rng.choice(sol.days))
    free = [truck for truck in sol.trucks if _truck_free(sol, truck, day, day + 1)]
    if not free:
        return None
    truck = int(rng.choice(free))
    car = int(rng.choice(sol.C))
    dealer = int(rng.choice(sol.dealers))
    trip = Trip(day, truck, car, [dealer], [0])

    def apply():
        _attach(sol, trip)

    def revert():
        _detach(sol, trip)

    apply()
    if not schedule_feasible(sol, truck):
        revert()
        return None
    return Proposal("add-dealer", ("add-dealer", trip.uid, dealer, day, truck), revert, apply)


def try_add_dealer(sol, rng):
    """Insert a dealer at quantity 0, on an existing tour or on a free truck."""
    if sol.trips and rng.random() < 0.5:
        proposal = _add_dealer_on_route(sol, rng)
        if proposal is not None:
            return proposal
    proposal = _add_dealer_on_free_truck(sol, rng)
    if proposal is not None or not sol.trips:
        return proposal
    return _add_dealer_on_route(sol, rng)


def try_remove_dealer(sol, rng):
    pool = [trip for trip in sol.trips if trip.stops]
    if not pool:
        return None
    trip = rng.choice(pool)
    index = rng.randrange(len(trip.stops))
    dealer = trip.stops[index]
    if len(trip.stops) == 1:
        def apply():
            _detach(sol, trip)

        def revert():
            _attach(sol, trip)

        apply()
        return Proposal(
            "remove-dealer",
            ("remove-dealer", trip.uid, dealer, trip.day),
            revert,
            apply,
        )
    saved = snapshot([trip])
    trip.stops.pop(index)
    trip.qty.pop(index)
    return finish(sol, [trip], saved, "remove-dealer",
                  ("remove-dealer", trip.uid, dealer, trip.day))


MOVE_NAMES = [
    "rebuild", "hand-over", "exchange", "add-car", "remove-car",
    "add-dealer", "remove-dealer",
]
_BUILDERS = (
    (try_rebuild, 1.0),
    (try_hand_over, 1.0),
    (try_exchange, 1.0),
    (try_add_car, 1.0),
    (try_remove_car, 1.0),
    (try_add_dealer, 1.0),
    (try_remove_dealer, 1.0),
)
_MOVE_FUNCS = [item[0] for item in _BUILDERS]
_MOVE_WEIGHTS = [item[1] for item in _BUILDERS]
_MOVE_STATS = {name: {"attempts": 0, "built": 0} for name in MOVE_NAMES}
_FUNC_NAME = {func: name for func, name in zip(_MOVE_FUNCS, MOVE_NAMES)}


def reset_move_stats():
    for name in MOVE_NAMES:
        _MOVE_STATS[name] = {"attempts": 0, "built": 0}


def move_stats():
    return {name: dict(_MOVE_STATS[name]) for name in MOVE_NAMES}


def set_move_weights(weights):
    if len(weights) != len(_MOVE_WEIGHTS):
        raise ValueError("expected one weight per move")
    _MOVE_WEIGHTS[:] = [float(value) for value in weights]


def current_move_weights():
    return list(_MOVE_WEIGHTS)


def calibrated_weights(cap=5.0):
    raw = []
    for name in MOVE_NAMES:
        attempts = _MOVE_STATS[name]["attempts"]
        built = _MOVE_STATS[name]["built"]
        rate = built / attempts if attempts else 0.0
        raw.append(cap if rate <= 1e-12 else 1.0 / rate)
    lowest = min(raw)
    scaled = [value / lowest for value in raw]
    ceiling = cap
    return [min(value, ceiling) for value in scaled]


def propose_move(sol, rng):
    for _ in range(80):
        builder = rng.choices(_MOVE_FUNCS, weights=_MOVE_WEIGHTS, k=1)[0]
        name = _FUNC_NAME[builder]
        _MOVE_STATS[name]["attempts"] += 1
        proposal = builder(sol, rng)
        if proposal is not None:
            _MOVE_STATS[name]["built"] += 1
            return proposal
    return None


def count_active(trips):
    return sum(1 for trip in trips if trip.stops and sum(q for q in trip.qty if q > 0) > 0)
