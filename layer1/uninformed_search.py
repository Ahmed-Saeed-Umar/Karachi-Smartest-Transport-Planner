"""
uninformed_search.py
--------------------
Implements three uninformed search algorithms for the Karachi transport graph:

  - BFS (Breadth-First Search)    : finds shortest path by number of hops
  - DFS (Depth-First Search)      : explores deep first, cycle detection included
  - UCS (Uniform Cost Search)     : finds cheapest path by travel time

All three share the same interface:
    algorithm(graph, start, goal) -> SearchResult

All three use Node and SearchResult from search_base.py.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from collections import deque
import heapq

from layer1.search_base import (
    SearchResult, Node, reconstruct_path, get_neighbors
)


# ---------------------------------------------------------------------------
# BFS  —  Breadth-First Search
# ---------------------------------------------------------------------------
# How it works:
#   Uses a FIFO queue (deque). Explores all nodes at depth 1 first,
#   then depth 2, then depth 3, and so on.
#
# Guarantees:
#   - Complete     : YES — will always find a path if one exists
#   - Optimal      : only if all edge weights are EQUAL (they aren't here)
#   - Time/Space   : O(b^d) where b=branching factor, d=depth of solution
#
# In our graph:
#   BFS finds the path with fewest STOPS, not the fastest travel time.
#   So BFS and UCS will often disagree on which path is "best".
# ---------------------------------------------------------------------------
def bfs(graph: dict, start: str, goal: str, heuristic=None) -> SearchResult:
    """
    Breadth-First Search.
    heuristic parameter is accepted but ignored (uninformed algorithm).
    """
    # --- initialise result tracking ---
    nodes_expanded = 0
    max_frontier   = 0

    # --- frontier: FIFO queue of Nodes ---
    frontier = deque()
    frontier.append(Node(state=start, parent=None, g_cost=0))

    # --- explored set: don't revisit nodes ---
    explored = set()

    while frontier:
        max_frontier = max(max_frontier, len(frontier))

        # Pop from the LEFT (oldest node first = breadth first)
        current = frontier.popleft()
        nodes_expanded += 1

        # --- goal check ---
        if current.state == goal:
            path = reconstruct_path(current)
            return SearchResult(
                algorithm      = "BFS",
                start          = start,
                goal           = goal,
                path           = path,
                cost           = current.g_cost,
                nodes_expanded = nodes_expanded,
                max_frontier   = max_frontier,
                found          = True
            )

        # --- skip already explored nodes ---
        if current.state in explored:
            continue
        explored.add(current.state)

        # --- expand: add unvisited neighbors to frontier ---
        for neighbor, weight in get_neighbors(graph, current.state):
            if neighbor not in explored:
                child = Node(
                    state   = neighbor,
                    parent  = current,
                    g_cost  = current.g_cost + weight  # tracked but not used for ordering
                )
                frontier.append(child)  # append to RIGHT = FIFO

    # No path found
    return SearchResult(
        algorithm="BFS", start=start, goal=goal,
        nodes_expanded=nodes_expanded, max_frontier=max_frontier, found=False
    )


# ---------------------------------------------------------------------------
# DFS  —  Depth-First Search  (with cycle detection)
# ---------------------------------------------------------------------------
# How it works:
#   Uses a LIFO stack. Dives deep into one branch before backtracking.
#
# Guarantees:
#   - Complete     : YES with cycle detection (without it, loops forever)
#   - Optimal      : NO — often finds a very bad path
#   - Space        : O(b*m) — much better than BFS for deep graphs
#
# In our graph:
#   DFS will likely take a winding route across the city before finding
#   the goal, giving a much higher cost than BFS or UCS.
#   Test case 5 is specifically chosen to expose this weakness.
# ---------------------------------------------------------------------------
def dfs(graph: dict, start: str, goal: str, heuristic=None) -> SearchResult:
    """
    Depth-First Search with cycle detection.
    heuristic parameter accepted but ignored.
    """
    nodes_expanded = 0
    max_frontier   = 0

    # --- frontier: LIFO stack of Nodes ---
    # We use a regular list and pop from the end
    frontier = []
    frontier.append(Node(state=start, parent=None, g_cost=0))

    # --- explored set ---
    explored = set()

    while frontier:
        max_frontier = max(max_frontier, len(frontier))

        # Pop from the RIGHT (most recently added = depth first)
        current = frontier.pop()
        nodes_expanded += 1

        # --- goal check ---
        if current.state == goal:
            path = reconstruct_path(current)
            return SearchResult(
                algorithm      = "DFS",
                start          = start,
                goal           = goal,
                path           = path,
                cost           = current.g_cost,
                nodes_expanded = nodes_expanded,
                max_frontier   = max_frontier,
                found          = True
            )

        # --- cycle detection: skip if already explored ---
        if current.state in explored:
            continue
        explored.add(current.state)

        # --- expand: push neighbors onto stack ---
        # Note: reversed so that neighbors are explored in their natural order
        for neighbor, weight in reversed(get_neighbors(graph, current.state)):
            if neighbor not in explored:
                child = Node(
                    state   = neighbor,
                    parent  = current,
                    g_cost  = current.g_cost + weight
                )
                frontier.append(child)

    return SearchResult(
        algorithm="DFS", start=start, goal=goal,
        nodes_expanded=nodes_expanded, max_frontier=max_frontier, found=False
    )


# ---------------------------------------------------------------------------
# UCS  —  Uniform Cost Search
# ---------------------------------------------------------------------------
# How it works:
#   Uses a MIN PRIORITY QUEUE ordered by g_cost (actual travel time so far).
#   Always expands the cheapest node in the frontier next.
#
# Guarantees:
#   - Complete     : YES
#   - Optimal      : YES — always finds the minimum-cost path
#   - Time/Space   : O(b^(1 + C*/e)) where C*=optimal cost, e=min edge cost
#
# In our graph:
#   UCS will find the fastest route in minutes, even if it has more stops.
#   This is the key difference from BFS which minimises stops, not time.
# ---------------------------------------------------------------------------
def ucs(graph: dict, start: str, goal: str, heuristic=None) -> SearchResult:
    """
    Uniform Cost Search.
    heuristic parameter accepted but ignored.
    """
    nodes_expanded = 0
    max_frontier   = 0

    # --- frontier: MIN priority queue ordered by (g_cost, node) ---
    # heapq in Python is a min-heap.
    # We push tuples: (priority, tie_breaker, Node)
    # tie_breaker is a counter to avoid comparing Node objects directly
    counter   = 0
    frontier  = []
    heapq.heappush(frontier, (0, counter, Node(state=start, parent=None, g_cost=0)))

    # --- explored set ---
    explored = set()

    while frontier:
        max_frontier = max(max_frontier, len(frontier))

        # Pop the node with the LOWEST g_cost
        cost, _, current = heapq.heappop(frontier)
        nodes_expanded += 1

        # --- goal check ---
        if current.state == goal:
            path = reconstruct_path(current)
            return SearchResult(
                algorithm      = "UCS",
                start          = start,
                goal           = goal,
                path           = path,
                cost           = current.g_cost,
                nodes_expanded = nodes_expanded,
                max_frontier   = max_frontier,
                found          = True
            )

        # --- skip if already explored with a cheaper path ---
        if current.state in explored:
            continue
        explored.add(current.state)

        # --- expand ---
        for neighbor, weight in get_neighbors(graph, current.state):
            if neighbor not in explored:
                new_cost = current.g_cost + weight
                counter += 1
                child = Node(
                    state   = neighbor,
                    parent  = current,
                    g_cost  = new_cost
                )
                heapq.heappush(frontier, (new_cost, counter, child))

    return SearchResult(
        algorithm="UCS", start=start, goal=goal,
        nodes_expanded=nodes_expanded, max_frontier=max_frontier, found=False
    )


# ---------------------------------------------------------------------------
# SANITY CHECK
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from data.khi_graph import build_graph
    from layer1.search_base import print_results_table

    graph = build_graph()

    test_pairs = [
        ("Saddar",  "Clifton"),     # short direct route
        ("Orangi",  "Korangi"),     # long cross-city
    ]

    for start, goal in test_pairs:
        results = [
            bfs(graph, start, goal),
            dfs(graph, start, goal),
            ucs(graph, start, goal),
        ]
        print_results_table(results)
        for r in results:
            print(f"  {r.algorithm}: {r.path_str()}")
        print()
