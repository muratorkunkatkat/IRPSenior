"""Simulated annealing on the full vehicle-distribution instance.

One neighbor is drawn per iteration. A cheaper neighbor is always kept.
A more expensive one is kept with probability exp(-delta / temperature).

T0 is set from the construction cost, not from a sample of random worsenings:
a plan 10% worse than the reference is accepted with a chosen probability
(75%, 50%, or 25%). The final temperature is the value at which a plan 1%
worse than the same reference is accepted with probability 0.0001.

Cooling is one of two rules. If `alpha` is set, temperature is multiplied by
alpha after every block of `cool_every` (L) scored neighbors, and is not
allowed below T_final. That is the paper schedule. If `alpha` is omitted and
a time limit is set, temperature follows the clock from T0 to T_final.
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


def paper_temperatures(reference):
    delta10 = 0.10 * reference
    delta1 = 0.01 * reference
    return {
        "reference": reference,
        "delta10": delta10,
        "delta1": delta1,
        "T75": -delta10 / math.log(0.75),
        "T50": -delta10 / math.log(0.50),
        "T25": -delta10 / math.log(0.25),
        "T_final": -delta1 / math.log(0.0001),
    }


def _accept(delta, temperature, rng):
    if delta <= 0:
        return True
    if temperature <= 1e-12:
        return False
    scaled = delta / temperature
    if scaled > 60.0:
        return False
    return rng.random() < math.exp(-scaled)


def _temperature(temperature0, t_final, elapsed, time_limit_sec):
    if time_limit_sec <= 0 or temperature0 <= t_final:
        return t_final
    progress = min(1.0, max(0.0, elapsed / time_limit_sec))
    return temperature0 * (t_final / temperature0) ** progress


def run_simulated_annealing(data, iterations=1_000_000, seed=44, trips=None,
                            t0=None, t_final=None, time_limit_sec=None,
                            log_every=None, alpha=None, cool_every=10,
                            history_every=1):
    rng = random.Random(seed)
    sol = Solution(data, clone_trips(trips) if trips is not None else build_initial_trips(data))
    initial, aux = evaluate(sol)
    sol.cost = initial
    sol.aux = aux

    temps = paper_temperatures(initial.total)
    temperature0 = temps["T50"] if t0 is None else float(t0)
    final_t = temps["T_final"] if t_final is None else float(t_final)
    started = time.perf_counter()
    temperature = temperature0

    best_cost = initial
    best_trips = clone_trips(sol.trips)
    best_iteration = 0
    current = initial
    move_counts = {}
    proposed_counts = {}
    rejected_counts = {}
    accepted_flags = []
    history = [{
        "iteration": 0,
        "current": initial.total,
        "best": initial.total,
        "best_transport": initial.transport,
        "best_backorder": initial.backorder,
        "best_holding": initial.holding,
        "best_unsold": initial.unsold,
        "seconds": 0.0,
        "temperature": temperature0,
        "accepted": False,
    }]

    if log_every is None:
        log_every = max(1, iterations // 10)

    ran = 0
    scored = 0
    stopped = "iterations"
    if alpha is not None:
        cool_note = f"alpha={alpha} every {cool_every} neighbors"
    elif time_limit_sec:
        cool_note = f"clock cool over {time_limit_sec:.0f}s"
    else:
        cool_note = "geometric over iterations"
    print(
        f"SA start  total={initial.total:,.2f}  T0={temperature0:,.2f}  "
        f"T_final={final_t:,.2f}  {cool_note}",
        flush=True,
    )

    for iteration in range(1, iterations + 1):
        elapsed_now = time.perf_counter() - started
        if time_limit_sec is not None and elapsed_now >= time_limit_sec:
            stopped = "time limit"
            break
        if alpha is None and time_limit_sec:
            temperature = _temperature(temperature0, final_t, elapsed_now, time_limit_sec)
        ran = iteration
        proposal = propose_move(sol, rng)
        accepted = False
        improved = False
        if proposal is not None:
            scored += 1
            proposed_counts[proposal.kind] = proposed_counts.get(proposal.kind, 0) + 1
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
                    improved = True
            else:
                proposal.undo()
                rejected_counts[proposal.kind] = rejected_counts.get(proposal.kind, 0) + 1
            if alpha is not None and scored % cool_every == 0:
                temperature = max(final_t, temperature * alpha)
        if history_every <= 1:
            accepted_flags.append(accepted)
        if alpha is None and not time_limit_sec:
            temperature = max(final_t, temperature * (final_t / temperature0) ** (1 / max(iterations, 1)))
        if history_every <= 1 or iteration % history_every == 0 or improved:
            history.append({
                "iteration": iteration,
                "current": current.total,
                "best": best_cost.total,
                "best_transport": best_cost.transport,
                "best_backorder": best_cost.backorder,
                "best_holding": best_cost.holding,
                "best_unsold": best_cost.unsold,
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
    if ran and history[-1]["iteration"] != ran:
        history.append({
            "iteration": ran,
            "current": current.total,
            "best": best_cost.total,
            "best_transport": best_cost.transport,
            "best_backorder": best_cost.backorder,
            "best_holding": best_cost.holding,
            "best_unsold": best_cost.unsold,
            "seconds": elapsed,
            "temperature": temperature,
            "accepted": False,
        })
    exact, _aux = evaluate(Solution(data, best_trips))
    print(
        f"SA done  best={exact.total:,.2f}  iter={best_iteration}  "
        f"seconds={elapsed:.1f}  stop={stopped}  T_end={temperature:,.2f}",
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
        "proposed_counts": proposed_counts,
        "rejected_counts": rejected_counts,
        "accepted_flags": accepted_flags,
        "stopped": stopped,
        "parameters": {
            "iterations": iterations,
            "seed": seed,
            "T0": temperature0,
            "T_final": final_t,
            "alpha": alpha,
            "cool_every": cool_every,
            "time_limit_sec": time_limit_sec,
            "temps": temps,
        },
    }
