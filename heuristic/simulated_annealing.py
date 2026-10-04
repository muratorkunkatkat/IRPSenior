"""Simulated annealing on the full vehicle-distribution instance.

One random neighbor is drawn per iteration. A cheaper neighbor is always kept.
A more expensive one is kept with probability exp(-delta / temperature).
Temperature cools geometrically from a value calibrated on the initial solution,
so early uphill moves are common and late ones are rare.
"""

import math
import random
import time

from heuristic.search_problem import (
    Solution,
    build_initial_trips,
    clone_trips,
    evaluate,
    propose_move,
)


def _accept(delta, temperature, rng):
    if delta <= 0:
        return True
    if temperature <= 1e-12:
        return False
    scaled = delta / temperature
    if scaled > 60.0:
        return False
    return rng.random() < math.exp(-scaled)


def _calibrate(sol, rng, samples):
    """Temperature at which a typical uphill move is still often accepted."""
    uphill = []
    base = sol.cost.total
    for _ in range(samples):
        proposal = propose_move(sol, rng)
        if proposal is None:
            continue
        cost, _aux = evaluate(sol)
        delta = cost.total - base
        proposal.undo()
        if delta > 1e-6:
            uphill.append(delta)
    if not uphill:
        return 1000.0
    uphill.sort()
    median = uphill[len(uphill) // 2]
    return max(median / -math.log(0.8), 1.0)


def run_simulated_annealing(data, iterations=800, seed=44, trips=None,
                            sample_moves=24, final_temp_ratio=0.001,
                            time_limit_sec=None, log_every=None):
    rng = random.Random(seed)
    sol = Solution(data, clone_trips(trips) if trips is not None else build_initial_trips(data))
    initial, aux = evaluate(sol)
    sol.cost = initial
    sol.aux = aux

    started = time.perf_counter()
    temperature0 = _calibrate(sol, rng, sample_moves)
    alpha = math.exp(math.log(final_temp_ratio) / max(iterations, 1))
    temperature = temperature0

    best_cost = initial
    best_trips = clone_trips(sol.trips)
    best_iteration = 0
    current = initial
    move_counts = {}
    accepted_flags = []
    history = [{
        "iteration": 0,
        "current": initial.total,
        "best": initial.total,
        "best_transport": initial.transport,
        "best_backorder": initial.backorder,
        "seconds": 0.0,
        "temperature": temperature0,
        "accepted": False,
    }]

    if log_every is None:
        log_every = max(1, iterations // 10)

    ran = 0
    stopped = "iterations"
    print(f"SA start  total={initial.total:,.2f}  T0={temperature0:,.2f}  alpha={alpha:.6f}", flush=True)

    for iteration in range(1, iterations + 1):
        if time_limit_sec is not None and (time.perf_counter() - started) >= time_limit_sec:
            stopped = "time limit"
            break
        ran = iteration
        proposal = propose_move(sol, rng)
        accepted = False
        if proposal is not None:
            trial, trial_aux = evaluate(sol)
            delta = trial.total - current.total
            if _accept(delta, temperature, rng):
                accepted = True
                current = trial
                sol.cost = trial
                sol.aux = trial_aux
                move_counts[proposal.kind] = move_counts.get(proposal.kind, 0) + 1
                if trial.total + 1e-6 < best_cost.total:
                    best_cost = trial
                    best_trips = clone_trips(sol.trips)
                    best_iteration = iteration
            else:
                proposal.undo()
        accepted_flags.append(accepted)
        temperature *= alpha
        history.append({
            "iteration": iteration,
            "current": current.total,
            "best": best_cost.total,
            "best_transport": best_cost.transport,
            "best_backorder": best_cost.backorder,
            "seconds": time.perf_counter() - started,
            "temperature": temperature,
            "accepted": accepted,
        })
        if iteration % log_every == 0 or iteration == iterations:
            print(
                f"SA  iter {iteration:>5}  current={current.total:,.2f}  "
                f"best={best_cost.total:,.2f}  T={temperature:,.2f}",
                flush=True,
            )

    elapsed = time.perf_counter() - started
    exact, _aux = evaluate(Solution(data, best_trips))
    print(
        f"SA done  best={exact.total:,.2f}  iter={best_iteration}  "
        f"seconds={elapsed:.1f}  stop={stopped}",
        flush=True,
    )
    return {
        "name": "Simulated annealing",
        "initial_cost": initial,
        "best_cost": exact,
        "best_trips": best_trips,
        "best_iteration": best_iteration,
        "history": history,
        "elapsed": elapsed,
        "iterations_run": ran,
        "move_counts": move_counts,
        "accepted_flags": accepted_flags,
        "stopped": stopped,
        "parameters": {
            "iterations": iterations,
            "seed": seed,
            "T0": temperature0,
            "alpha": alpha,
            "final_temp_ratio": final_temp_ratio,
            "time_limit_sec": time_limit_sec,
        },
    }
