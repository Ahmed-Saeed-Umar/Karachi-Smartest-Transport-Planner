"""
informed_search.py
------------------
Implements three informed search algorithms:

  - Greedy Best-First Search : uses only h(n), fast but not optimal
  - A*                       : uses f(n) = g(n) + h(n), optimal + complete
  - IDA*                     : iterative deepening A*, memory-bounded variant

Why IDA* over RBFS or SMA*?
  Our Karachi graph has 22 nodes — relatively small. IDA* is the right
  choice because:
    - It uses O(d) memory (just the current path) vs O(b^d) for A*
    - It's simpler to implement correctly than RBFS or SMA*
    - On small graphs it doesn't suffer much from repeated work
    - RBFS and SMA* shine on very large graphs where frontier itself
      is the memory problem — not the case here

All three accept a heuristic as a parameter (h1 or h2 from heuristics.py).
Same interface as Layer 1: (graph, start, goal, heuristic) -> SearchResult
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import heapq
import math

from layer1.search_base import (
    SearchResult, Node, reconstruct_path, get_neighbors
)


# ---------------------------------------------------------------------------
# GREEDY BEST-FIRST SEARCH
# ---------------------------------------------------------------------------
# How it works:
#   Priority queue ordered by h(n) ONLY — ignores actual cost g(n).
#   Always expands the node that looks closest to the goal.
#
# Guarantees:
#   - Complete : YES (with cycle detection)
#   - Optimal  : NO  — ignores cost already paid, can find bad paths
#
# In our graph:
#   Greedy will zoom toward the goal geographically but may miss a
#   cheaper route that goes slightly away first. Test case 3 (Orangi->DHA)
#   is a good example where it gets tricked by the creek bottleneck.
# ---------------------------------------------------------------------------
def greedy(graph: dict, start: str, goal: str, heuristic) -> SearchResult:
    """Greedy Best-First Search — ordered by h(n) only."""
    nodes_expanded = 0
    max_frontier   = 0
    counter        = 0

    start_h = heuristic(start, goal)
    frontier = []
    heapq.heappush(frontier, (start_h, counter,
                               Node(state=start, parent=None,
                                    g_cost=0, h_cost=start_h)))
    explored = set()

    while frontier:
        max_frontier = max(max_frontier, len(frontier))

        _, _, current = heapq.heappop(frontier)
        nodes_expanded += 1

        if current.state == goal:
            path = reconstruct_path(current)
            return SearchResult(
                algorithm      = "Greedy",
                start          = start,
                goal           = goal,
                path           = path,
                cost           = current.g_cost,
                nodes_expanded = nodes_expanded,
                max_frontier   = max_frontier,
                found          = True
            )

        if current.state in explored:
            continue
        explored.add(current.state)

        for neighbor, weight in get_neighbors(graph, current.state):
            if neighbor not in explored:
                h = heuristic(neighbor, goal)
                counter += 1
                child = Node(
                    state   = neighbor,
                    parent  = current,
                    g_cost  = current.g_cost + weight,
                    h_cost  = h
                )
                # Greedy orders by h(n) only — ignores g_cost
                heapq.heappush(frontier, (h, counter, child))

    return SearchResult(
        algorithm="Greedy", start=start, goal=goal,
        nodes_expanded=nodes_expanded, max_frontier=max_frontier, found=False
    )


# ---------------------------------------------------------------------------
# A* SEARCH
# ---------------------------------------------------------------------------
# How it works:
#   Priority queue ordered by f(n) = g(n) + h(n).
#   Balances actual cost paid and estimated cost remaining.
#
# Guarantees:
#   - Complete : YES
#   - Optimal  : YES — if heuristic is admissible
#   - With consistent heuristic: never re-expands a node
#
# In our graph:
#   A* with h2 should expand significantly fewer nodes than UCS
#   while still finding the optimal path. This is the key result
#   for the Layer 2 report.
# ---------------------------------------------------------------------------
def astar(graph: dict, start: str, goal: str, heuristic) -> SearchResult:
    """A* Search — ordered by f(n) = g(n) + h(n)."""
    nodes_expanded = 0
    max_frontier   = 0
    counter        = 0

    start_h = heuristic(start, goal)
    frontier = []
    heapq.heappush(frontier, (start_h, counter,
                               Node(state=start, parent=None,
                                    g_cost=0, h_cost=start_h)))
    explored = set()

    while frontier:
        max_frontier = max(max_frontier, len(frontier))

        _, _, current = heapq.heappop(frontier)
        nodes_expanded += 1

        if current.state == goal:
            path = reconstruct_path(current)
            return SearchResult(
                algorithm      = "A*",
                start          = start,
                goal           = goal,
                path           = path,
                cost           = current.g_cost,
                nodes_expanded = nodes_expanded,
                max_frontier   = max_frontier,
                found          = True
            )

        if current.state in explored:
            continue
        explored.add(current.state)

        for neighbor, weight in get_neighbors(graph, current.state):
            if neighbor not in explored:
                g = current.g_cost + weight
                h = heuristic(neighbor, goal)
                counter += 1
                child = Node(
                    state   = neighbor,
                    parent  = current,
                    g_cost  = g,
                    h_cost  = h
                )
                # A* orders by f = g + h
                heapq.heappush(frontier, (g + h, counter, child))

    return SearchResult(
        algorithm="A*", start=start, goal=goal,
        nodes_expanded=nodes_expanded, max_frontier=max_frontier, found=False
    )


# ---------------------------------------------------------------------------
# IDA* — Iterative Deepening A*
# ---------------------------------------------------------------------------
# How it works:
#   Runs depth-first searches with increasing f-cost thresholds.
#   Each iteration explores all nodes with f(n) <= threshold.
#   If goal not found, threshold increases to the smallest f-value
#   that exceeded the current threshold.
#
# Guarantees:
#   - Complete : YES
#   - Optimal  : YES — same as A* with admissible heuristic
#   - Memory   : O(d) — only stores current path, not entire frontier
#
# Why it suits our graph:
#   22 nodes means memory isn't a crisis, but IDA* still demonstrates
#   the memory-bounded concept clearly. Its nodes_expanded count will
#   be higher than A* due to repeated work, but max_frontier = path depth.
# ---------------------------------------------------------------------------
def idastar(graph: dict, start: str, goal: str, heuristic) -> SearchResult:
    """IDA* — Iterative Deepening A*."""

    nodes_expanded = 0
    max_frontier   = 0  # IDA* frontier = current path length

    def search(path: list, g: float, threshold: float):
        """
        Recursive DFS within the current f-cost threshold.

        Returns:
          - float  : new threshold if goal not found (min f that exceeded threshold)
          - 'FOUND': if goal is reached
        """
        nonlocal nodes_expanded, max_frontier

        current_node = path[-1]
        f = g + heuristic(current_node.state, goal)

        # Exceeded threshold — return f as the next candidate threshold
        if f > threshold:
            return f

        nodes_expanded += 1
        max_frontier = max(max_frontier, len(path))

        # Goal check
        if current_node.state == goal:
            return "FOUND"

        minimum = math.inf

        for neighbor, weight in get_neighbors(graph, current_node.state):
            # Cycle check: don't revisit nodes already on current path
            if any(n.state == neighbor for n in path):
                continue

            child = Node(
                state   = neighbor,
                parent  = current_node,
                g_cost  = g + weight,
                h_cost  = heuristic(neighbor, goal)
            )
            path.append(child)
            result = search(path, g + weight, threshold)

            if result == "FOUND":
                return "FOUND"
            if result < minimum:
                minimum = result

            path.pop()

        return minimum

    # --- IDA* main loop ---
    start_node = Node(state=start, parent=None, g_cost=0,
                      h_cost=heuristic(start, goal))
    threshold  = heuristic(start, goal)
    path       = [start_node]

    while True:
        result = search(path, 0, threshold)

        if result == "FOUND":
            full_path = [n.state for n in path]
            total_cost = path[-1].g_cost
            return SearchResult(
                algorithm      = "IDA*",
                start          = start,
                goal           = goal,
                path           = full_path,
                cost           = total_cost,
                nodes_expanded = nodes_expanded,
                max_frontier   = max_frontier,
                found          = True
            )

        if result == math.inf:
            # No path exists
            return SearchResult(
                algorithm="IDA*", start=start, goal=goal,
                nodes_expanded=nodes_expanded,
                max_frontier=max_frontier, found=False
            )

        # Raise threshold to the next candidate value
        threshold = result


# ---------------------------------------------------------------------------
# SANITY CHECK
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

    from data.khi_graph import build_graph
    from layer1.search_base import print_results_table
    from layer2.heuristics import h1, h2

    graph = build_graph()

    test_pairs = [
        ("Saddar",  "Korangi"),
        ("Orangi",  "DHA"),
        ("Surjani", "Kemari"),
    ]

    for start, goal in test_pairs:
        results = [
            greedy(graph, start, goal, h2),
            astar(graph,  start, goal, h2),
            idastar(graph, start, goal, h2),
        ]
        print_results_table(results)
        for r in results:
            print(f"  {r.algorithm:6s}: {r.path_str()}")
        print()
