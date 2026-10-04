"""Tabu search on the full vehicle-distribution instance.

Each iteration scores a sample of random neighbors and steps to the cheapest
admissible one. A move whose attribute is still in the tabu list is skipped,
unless it beats the best solution seen so far (aspiration). If every sampled
move is tabu, the cheapest of them is taken anyway so the search does not stall.
"""

import random
import time

from heuristic.search_problem import (
    Solution,
    build_initial_trips,
    clone_trips,
    evaluate,
    propose_move,
)


def run_tabu_search(data, iterations=250, neighbors=16, tenure=12, seed=44,
                    trips=None, time_limit_sec=None, log_every=None):
    rng = random.Random(seed)
    sol = Solution(data, clone_trips(trips) if trips is not None else build_initial_trips(data))
    initial, aux = evaluate(sol)
    sol.cost = initial
    sol.aux = aux

    best_cost = initial
    best_trips = clone_trips(sol.trips)
    best_iteration = 0
    current = initial
    tabu_until = {}
    move_counts = {}
    history = [{
        "iteration": 0,
        "current": initial.total,
        "best": initial.total,
        "best_transport": initial.transport,
        "best_backorder": initial.backorder,
        "seconds": 0.0,
    }]

    if log_every is None:
        log_every = max(1, iterations // 10)

    started = time.perf_counter()
    ran = 0
    stopped = "iterations"
    print(
        f"Tabu start  total={initial.total:,.2f}  neighbors={neighbors}  tenure={tenure}",
        flush=True,
    )

    for iteration in range(1, iterations + 1):
        if time_limit_sec is not None and (time.perf_counter() - started) >= time_limit_sec:
            stopped = "time limit"
            break
        ran = iteration
        candidates = []
        attempts = 0
        while len(candidates) < neighbors and attempts < neighbors * 5:
            attempts += 1
            proposal = propose_move(sol, rng)
            if proposal is None:
                continue
            trial, trial_aux = evaluate(sol)
            proposal.undo()
            candidates.append((trial, trial_aux, proposal))

        if not candidates:
            history.append({
                "iteration": iteration,
                "current": current.total,
                "best": best_cost.total,
                "best_transport": best_cost.transport,
                "best_backorder": best_cost.backorder,
                "seconds": time.perf_counter() - started,
            })
            continue

        admissible = []
        for trial, trial_aux, proposal in candidates:
            expires = tabu_until.get(proposal.key, -1)
            is_tabu = expires >= iteration
            if (not is_tabu) or trial.total + 1e-6 < best_cost.total:
                admissible.append((trial, trial_aux, proposal))
        pool = admissible if admissible else candidates
        trial, trial_aux, proposal = min(pool, key=lambda item: item[0].total)
        proposal.redo()
        sol.cost = trial
        sol.aux = trial_aux
        current = trial
        tabu_until[proposal.key] = iteration + tenure
        move_counts[proposal.kind] = move_counts.get(proposal.kind, 0) + 1
        if trial.total + 1e-6 < best_cost.total:
            best_cost = trial
            best_trips = clone_trips(sol.trips)
            best_iteration = iteration

        history.append({
            "iteration": iteration,
            "current": current.total,
            "best": best_cost.total,
            "best_transport": best_cost.transport,
            "best_backorder": best_cost.backorder,
            "seconds": time.perf_counter() - started,
        })
        if iteration % log_every == 0 or iteration == iterations:
            print(
                f"Tabu  iter {iteration:>5}  current={current.total:,.2f}  "
                f"best={best_cost.total:,.2f}",
                flush=True,
            )

    elapsed = time.perf_counter() - started
    exact, _aux = evaluate(Solution(data, best_trips))
    print(
        f"Tabu done  best={exact.total:,.2f}  iter={best_iteration}  "
        f"seconds={elapsed:.1f}  stop={stopped}",
        flush=True,
    )
    return {
        "name": "Tabu search",
        "initial_cost": initial,
        "best_cost": exact,
        "best_trips": best_trips,
        "best_iteration": best_iteration,
        "history": history,
        "elapsed": elapsed,
        "iterations_run": ran,
        "move_counts": move_counts,
        "stopped": stopped,
        "parameters": {
            "iterations": iterations,
            "neighbors": neighbors,
            "tenure": tenure,
            "seed": seed,
            "time_limit_sec": time_limit_sec,
        },
    }
