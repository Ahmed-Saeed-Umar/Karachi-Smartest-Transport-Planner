"""
local_search.py
---------------
Three local search algorithms for the Karachi network optimisation problem:

  1. Hill Climbing         — steepest ascent, with and without sideways moves
  2. Random-Restart HC     — runs Hill Climbing from multiple random starts
  3. Simulated Annealing   — accepts worse moves probabilistically to escape local optima

All algorithms maximise the objective function from objective_function.py.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import random
import math
from layer3.objective_func import (
    NetworkState, objective, get_neighbours, random_state,
    K_STOPS, MAX_ROUTES
)


# ---------------------------------------------------------------------------
# 1. HILL CLIMBING  — steepest ascent
# ---------------------------------------------------------------------------
# How it works:
#   From the current state, evaluate ALL neighbours.
#   Move to the best neighbour if it's better than current.
#   Stop when no neighbour is better (local maximum reached).
#
# With sideways moves:
#   Also accept neighbours with equal score (up to a limit).
#   This helps escape flat plateaus but can cause infinite loops
#   if the plateau is very large — hence the sideways move limit.
# ---------------------------------------------------------------------------
def hill_climbing(graph: dict,
                  initial_state: NetworkState = None,
                  allow_sideways: bool = False,
                  max_sideways: int = 20,
                  max_steps: int = 1000) -> dict:
    """
    Steepest-ascent Hill Climbing.

    Parameters
    ----------
    graph          : Karachi transport graph
    initial_state  : starting NetworkState (random if None)
    allow_sideways : whether to accept equal-score moves
    max_sideways   : max consecutive sideways moves allowed
    max_steps      : hard stop to prevent infinite loops

    Returns
    -------
    dict with keys: state, score, steps, sideways_taken, stuck_reason
    """
    if initial_state is None:
        initial_state = random_state(graph)

    current       = initial_state
    current_score = objective(current, graph)
    steps         = 0
    sideways_taken = 0

    for _ in range(max_steps):
        neighbours = get_neighbours(current, graph)
        if not neighbours:
            break

        # Find best neighbour
        best_nb    = None
        best_score = current_score

        for nb in neighbours:
            s = objective(nb, graph)
            if s > best_score:
                best_score = s
                best_nb    = nb

        # Strict improvement found
        if best_nb is not None:
            current        = best_nb
            current_score  = best_score
            sideways_taken = 0
            steps         += 1
            continue

        # No improvement — check for sideways move
        if allow_sideways and sideways_taken < max_sideways:
            # Find any neighbour with equal score
            equal_nbs = [nb for nb in neighbours
                         if abs(objective(nb, graph) - current_score) < 1e-6]
            if equal_nbs:
                current        = random.choice(equal_nbs)
                sideways_taken += 1
                steps          += 1
                continue

        # Stuck at local maximum
        break

    stuck_reason = "local_max" if steps < max_steps else "max_steps"

    return {
        "state"          : current,
        "score"          : current_score,
        "steps"          : steps,
        "sideways_taken" : sideways_taken,
        "stuck_reason"   : stuck_reason,
    }


# ---------------------------------------------------------------------------
# 2. RANDOM-RESTART HILL CLIMBING
# ---------------------------------------------------------------------------
# How it works:
#   Run Hill Climbing from N random starting states.
#   Keep the best result found across all runs.
#
# Why it helps:
#   Hill Climbing always gets stuck at a local maximum.
#   Restarting from random positions explores different basins of
#   attraction in the solution space, increasing the chance of
#   finding the global maximum.
# ---------------------------------------------------------------------------
def random_restart_hc(graph: dict,
                      n_restarts: int = 50,
                      allow_sideways: bool = False,
                      max_sideways: int = 20) -> dict:
    """
    Random-Restart Hill Climbing.

    Runs hill_climbing n_restarts times from random initial states.
    Returns the best result found, plus statistics across all runs.
    """
    best_result    = None
    all_scores     = []
    all_steps      = []
    success_count  = 0   # runs that improved over their starting score

    for i in range(n_restarts):
        init  = random_state(graph)
        init_score = objective(init, graph)

        result = hill_climbing(
            graph,
            initial_state  = init,
            allow_sideways = allow_sideways,
            max_sideways   = max_sideways
        )

        all_scores.append(result["score"])
        all_steps.append(result["steps"])

        if result["score"] > init_score:
            success_count += 1

        if best_result is None or result["score"] > best_result["score"]:
            best_result = result

    return {
        "best_state"    : best_result["state"],
        "best_score"    : best_result["score"],
        "avg_score"     : sum(all_scores) / len(all_scores),
        "avg_steps"     : sum(all_steps)  / len(all_steps),
        "success_rate"  : success_count / n_restarts,
        "all_scores"    : all_scores,
        "n_restarts"    : n_restarts,
    }


# ---------------------------------------------------------------------------
# 3. SIMULATED ANNEALING
# ---------------------------------------------------------------------------
# How it works:
#   Like Hill Climbing but sometimes accepts WORSE moves.
#   The probability of accepting a worse move is:
#       P = exp(delta / T)
#   where delta = new_score - current_score (negative for worse moves)
#   and T is the current "temperature".
#
#   Temperature starts high (accepts many bad moves = exploration)
#   and cools down over time (accepts fewer bad moves = exploitation).
#
# Cooling schedule:
#   T(t) = T_initial * r^t   where r is the cooling rate (0 < r < 1)
#   Higher r = slower cooling = more exploration time
# ---------------------------------------------------------------------------
def simulated_annealing(graph: dict,
                        T_initial: float = 10.0,
                        cooling_rate: float = 0.95,
                        T_min: float = 0.01,
                        max_steps: int = 5000,
                        initial_state: NetworkState = None) -> dict:
    """
    Simulated Annealing.

    Parameters
    ----------
    graph         : Karachi transport graph
    T_initial     : starting temperature
    cooling_rate  : r in T *= r each step (try 0.90, 0.95, 0.99)
    T_min         : stop when temperature falls below this
    max_steps     : hard stop
    initial_state : starting NetworkState (random if None)

    Returns
    -------
    dict with keys: state, score, steps, T_final,
                    score_history (every 100 steps)
    """
    if initial_state is None:
        initial_state = random_state(graph)

    current       = initial_state
    current_score = objective(current, graph)

    # Track best ever seen (SA can move away from it)
    best_state = current.copy()
    best_score = current_score

    T             = T_initial
    steps         = 0
    score_history = []

    while T > T_min and steps < max_steps:
        # Pick a RANDOM neighbour (not the best — that's the SA difference)
        neighbours = get_neighbours(current, graph)
        if not neighbours:
            break
        neighbour   = random.choice(neighbours)
        nb_score    = objective(neighbour, graph)

        delta = nb_score - current_score

        # Always accept improvements; accept worse moves with probability e^(delta/T)
        if delta > 0 or random.random() < math.exp(delta / T):
            current       = neighbour
            current_score = nb_score

            # Update best
            if current_score > best_score:
                best_score = current_score
                best_state = current.copy()

        # Cool down
        T     *= cooling_rate
        steps += 1

        # Record history every 100 steps
        if steps % 100 == 0:
            score_history.append((steps, current_score, T))

    return {
        "state"         : best_state,
        "score"         : best_score,
        "steps"         : steps,
        "T_final"       : T,
        "score_history" : score_history,
    }


# ---------------------------------------------------------------------------
# SANITY CHECK
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    random.seed(42)
    from data.khi_graph import build_graph

    graph = build_graph()

    print("="*60)
    print("Hill Climbing (no sideways)")
    result = hill_climbing(graph)
    print(f"  Score : {result['score']:.2f}")
    print(f"  Steps : {result['steps']}")
    print(f"  Stops : {result['state'].active_stops}")

    print("\nHill Climbing (with sideways)")
    result = hill_climbing(graph, allow_sideways=True)
    print(f"  Score : {result['score']:.2f}")
    print(f"  Steps : {result['steps']}")

    print("\nRandom-Restart HC (10 restarts)")
    result = random_restart_hc(graph, n_restarts=10)
    print(f"  Best score  : {result['best_score']:.2f}")
    print(f"  Avg score   : {result['avg_score']:.2f}")
    print(f"  Success rate: {result['success_rate']:.0%}")

    print("\nSimulated Annealing (r=0.95)")
    result = simulated_annealing(graph, cooling_rate=0.95)
    print(f"  Best score : {result['score']:.2f}")
    print(f"  Steps      : {result['steps']}")
    print(f"  T_final    : {result['T_final']:.4f}")
