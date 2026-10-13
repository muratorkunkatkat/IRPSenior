"""Shared solution, cost, and neighborhood for the large-instance metaheuristics.

Both searches start from the same dynamic-greedy plan and score it with the same
function, so a difference in the plots is a difference in the search, not in the
accounting.

Objective = transport + daily holding + daily backorder
            + leftover stock after the last day + dealer overflow penalty.

Transport uses R. Holding uses H on positive end-of-day stock. Backorder uses B
on the shortage carried that day. Leftover stock after the horizon is charged at
G per car. Stock above a dealer's storage limit is charged at G per car per day.

The neighborhood is five driving decisions. Every truck leaves the Bursa factory
and returns there. A neighbor never moves a delivery onto another day and never
changes how many cars a dealer gets:

  exchange    Two trucks of the same model, same day: trade one dealer each.
  hand-over   One dealer moves onto another same-day truck that still has room.
  resequence  Same dealers on the same truck; rebuild the visit order by distance.
  swap        On one tour, exchange the visit index of two dealers
              (0-1-2-3-0 becomes 0-2-1-3-0).
  shift       On one tour, rotate the dealer list by one toward the factory return
              (0-1-2-3-0 becomes 0-3-1-2-0).
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


# Share of proposals that are allowed to look more expensive on kilometers.
# Annealing then applies the temperature test. Tabu usually skips them,
# because it keeps the cheapest candidate in its sample.
UPHILL_SHARE = 0.2


def _active(sol):
    return [trip for trip in sol.trips if trip.stops and sum(trip.qty) > 0]


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


def _nn_order(R, stops):
    """Visit order that always leaves Bursa and returns there, by transport cost."""
    remaining = list(stops)
    order = []
    current = 0
    while remaining:
        best_i = 0
        best_cost = R[current][remaining[0]]
        for index in range(1, len(remaining)):
            cost = R[current][remaining[index]]
            if cost < best_cost:
                best_cost = cost
                best_i = index
        current = remaining.pop(best_i)
        order.append(current)
    return order


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


def _best_reinsertion(R, stops):
    """Move one stop to another place on the same tour, if that shortens it."""
    if len(stops) < 2:
        return None
    best = None
    best_cost = _path_cost(R, stops)
    for index, dealer in enumerate(stops):
        reduced = stops[:index] + stops[index + 1:]
        pos, _extra = _insertion_extra(R, reduced, dealer)
        trial = reduced[:pos] + [dealer] + reduced[pos:]
        if trial == stops:
            continue
        cost = _path_cost(R, trial)
        if cost + 1e-6 < best_cost:
            best_cost = cost
            best = trial
    return best


def _apply_order(trip, order, qty_of):
    trip.stops = list(order)
    trip.qty = [qty_of[stop] for stop in order]


def _same_day_groups(sol):
    groups = defaultdict(list)
    for trip in _active(sol):
        groups[(trip.day, trip.car_type)].append(trip)
    return groups


def try_resequence(sol, rng):
    """Same dealers, same cars, cheaper driving order back to Bursa."""
    pool = [trip for trip in _active(sol) if len(trip.stops) >= 2]
    if not pool:
        return None
    trip = rng.choice(pool)
    R = sol.R
    current = trip.stops
    current_cost = _path_cost(R, current)
    nearest = _nn_order(R, current)
    chosen = None
    if nearest != current and _path_cost(R, nearest) + 1e-6 < current_cost:
        chosen = nearest
    if chosen is None:
        chosen = _best_reinsertion(R, current)
    if chosen is None and nearest != current and rng.random() < UPHILL_SHARE:
        chosen = nearest
    if chosen is None:
        return None
    qty_of = {stop: qty for stop, qty in zip(trip.stops, trip.qty)}
    saved = snapshot([trip])
    _apply_order(trip, chosen, qty_of)
    return finish(sol, [trip], saved, "resequence", ("resequence", trip.uid))


def try_hand_over(sol, rng):
    """Move one dealer onto another same-day truck of the same model."""
    groups = _same_day_groups(sol)
    keys = [key for key, trips in groups.items() if len(trips) >= 2]
    if not keys:
        return None
    rng.shuffle(keys)
    R = sol.R
    for day, car in keys[:6]:
        trips = groups[(day, car)]
        src = rng.choice(trips)
        index = max(range(len(src.stops)), key=lambda i: _removal_saving(R, src.stops, i))
        if rng.random() < UPHILL_SHARE:
            index = rng.randrange(len(src.stops))
        dealer = src.stops[index]
        qty = src.qty[index]
        saving = _removal_saving(R, src.stops, index)
        receivers = [trip for trip in trips
                     if trip is not src and sum(trip.qty) + qty <= sol.Q[car]]
        if len(receivers) > 8:
            receivers = rng.sample(receivers, 8)
        best = None
        best_extra = None
        for dst in receivers:
            if dealer in dst.stops:
                extra = 0.0
                pos = None
            else:
                pos, extra = _insertion_extra(R, dst.stops, dealer)
            if best_extra is None or extra < best_extra:
                best_extra = extra
                best = (dst, pos)
        if best is None:
            continue
        if best_extra + 1e-6 >= saving and rng.random() >= UPHILL_SHARE:
            continue
        dst, pos = best
        saved = snapshot([src, dst])
        src.stops.pop(index)
        src.qty.pop(index)
        if dealer in dst.stops:
            dst.qty[dst.stops.index(dealer)] += qty
        else:
            dst.stops.insert(pos, dealer)
            dst.qty.insert(pos, qty)
        pair = (min(src.uid, dst.uid), max(src.uid, dst.uid))
        proposal = finish(sol, [src, dst], saved, "hand-over", ("hand-over", dealer, pair))
        if proposal is not None:
            return proposal
    return None


def try_exchange(sol, rng):
    """Swap one dealer between two same-day trucks of the same model.

    Hand-over needs an empty slot. Most tours leave full, so the useful
    decision is to trade two stops that are each a detour on the wrong truck.
    """
    groups = _same_day_groups(sol)
    keys = [key for key, trips in groups.items() if len(trips) >= 2]
    if not keys:
        return None
    rng.shuffle(keys)
    R = sol.R
    for _day, car in keys[:4]:
        trips = groups[(_day, car)]
        src, dst = rng.sample(trips, 2)
        src_idx = sorted(range(len(src.stops)),
                         key=lambda i: _removal_saving(R, src.stops, i), reverse=True)
        dst_idx = sorted(range(len(dst.stops)),
                         key=lambda i: _removal_saving(R, dst.stops, i), reverse=True)
        pairs = [(i, j) for i in src_idx[:2] for j in dst_idx[:2]]
        for _ in range(4):
            pairs.append((rng.randrange(len(src.stops)), rng.randrange(len(dst.stops))))
        best = None
        best_gain = None
        capacity = sol.Q[car]
        src_load = sum(src.qty)
        dst_load = sum(dst.qty)
        for i, j in pairs:
            dealer_a = src.stops[i]
            dealer_b = dst.stops[j]
            if dealer_a == dealer_b or dealer_a in dst.stops or dealer_b in src.stops:
                continue
            qty_a = src.qty[i]
            qty_b = dst.qty[j]
            if src_load - qty_a + qty_b > capacity or dst_load - qty_b + qty_a > capacity:
                continue
            src_rest = src.stops[:i] + src.stops[i + 1:]
            dst_rest = dst.stops[:j] + dst.stops[j + 1:]
            pos_b, _extra_b = _insertion_extra(R, src_rest, dealer_b)
            pos_a, _extra_a = _insertion_extra(R, dst_rest, dealer_a)
            trial_src = src_rest[:pos_b] + [dealer_b] + src_rest[pos_b:]
            trial_dst = dst_rest[:pos_a] + [dealer_a] + dst_rest[pos_a:]
            gain = (_path_cost(R, src.stops) + _path_cost(R, dst.stops)
                    - _path_cost(R, trial_src) - _path_cost(R, trial_dst))
            if best_gain is None or gain > best_gain:
                best_gain = gain
                best = (trial_src, trial_dst, dealer_a, dealer_b, qty_a, qty_b)
        if best is None:
            continue
        if best_gain <= 1e-6 and rng.random() >= UPHILL_SHARE:
            continue
        trial_src, trial_dst, dealer_a, dealer_b, qty_a, qty_b = best
        qty_src = {stop: qty for stop, qty in zip(src.stops, src.qty) if stop != dealer_a}
        qty_dst = {stop: qty for stop, qty in zip(dst.stops, dst.qty) if stop != dealer_b}
        qty_src[dealer_b] = qty_b
        qty_dst[dealer_a] = qty_a
        saved = snapshot([src, dst])
        _apply_order(src, trial_src, qty_src)
        _apply_order(dst, trial_dst, qty_dst)
        dealers = tuple(sorted((dealer_a, dealer_b)))
        tours = (min(src.uid, dst.uid), max(src.uid, dst.uid))
        proposal = finish(sol, [src, dst], saved, "exchange", ("exchange", dealers, tours))
        if proposal is not None:
            return proposal
    return None


def try_swap(sol, rng):
    """Exchange the visit index of two dealers on the same tour.

    Example: 0-1-2-3-0 becomes 0-2-1-3-0. The factory stays first and last;
    only the middle order changes.
    """
    pool = [trip for trip in _active(sol) if len(trip.stops) >= 2]
    if not pool:
        return None
    trip = rng.choice(pool)
    i, j = rng.sample(range(len(trip.stops)), 2)
    saved = snapshot([trip])
    trip.stops[i], trip.stops[j] = trip.stops[j], trip.stops[i]
    trip.qty[i], trip.qty[j] = trip.qty[j], trip.qty[i]
    a, b = sorted((trip.stops[i], trip.stops[j]))
    return finish(sol, [trip], saved, "swap", ("swap", trip.uid, a, b))


def try_shift(sol, rng):
    """Rotate the dealer list by one: last stop moves next to the factory leave.

    Example: 0-1-2-3-0 becomes 0-3-1-2-0. The truck still returns to Bursa;
    every dealer is still visited once.
    """
    pool = [trip for trip in _active(sol) if len(trip.stops) >= 2]
    if not pool:
        return None
    trip = rng.choice(pool)
    saved = snapshot([trip])
    trip.stops = trip.stops[-1:] + trip.stops[:-1]
    trip.qty = trip.qty[-1:] + trip.qty[:-1]
    return finish(sol, [trip], saved, "shift", ("shift", trip.uid, trip.stops[0]))


_BUILDERS = (
    (try_exchange, 2.5),
    (try_hand_over, 2.5),
    (try_resequence, 2.0),
    (try_swap, 2.0),
    (try_shift, 2.0),
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
