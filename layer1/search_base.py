"""
search_base.py
--------------
Shared interface and data structures for all search algorithms.

Every algorithm (BFS, DFS, UCS, Greedy, A*, IDA*) must:
  - Accept the same inputs  : graph, start, goal, [heuristic]
  - Return the same output  : a SearchResult object

This makes the comparison table in the report trivial to generate —
just call each algorithm and read off the same fields.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Callable


# ---------------------------------------------------------------------------
# SEARCH RESULT  — what every algorithm returns
# ---------------------------------------------------------------------------
@dataclass
class SearchResult:
    """
    Standardised output for every search algorithm.

    Fields
    ------
    algorithm       : name of the algorithm (e.g. "BFS", "A*")
    start           : starting node
    goal            : goal node
    path            : list of nodes from start to goal, [] if not found
    cost            : total travel time in minutes along the path
    nodes_expanded  : how many nodes were popped from the frontier
    max_frontier    : peak size of the frontier during search
    found           : True if a path was found
    """
    algorithm       : str
    start           : str
    goal            : str
    path            : List[str]     = field(default_factory=list)
    cost            : float         = 0.0
    nodes_expanded  : int           = 0
    max_frontier    : int           = 0
    found           : bool          = False

    def path_str(self) -> str:
        """Return path as a readable arrow string, e.g. Saddar -> Gulberg -> Johar"""
        return " -> ".join(self.path) if self.path else "No path found"

    def summary(self) -> str:
        """One-line summary for quick printing."""
        if self.found:
            return (f"[{self.algorithm}] {self.start} -> {self.goal} | "
                    f"Cost: {self.cost:.0f} min | "
                    f"Expanded: {self.nodes_expanded} | "
                    f"Max frontier: {self.max_frontier}")
        else:
            return f"[{self.algorithm}] {self.start} -> {self.goal} | NO PATH FOUND"


# ---------------------------------------------------------------------------
# NODE  — what sits inside the frontier during search
# ---------------------------------------------------------------------------
@dataclass
class Node:
    """
    A search node in the frontier.

    state    : current location name
    parent   : the Node we came from (None for the start node)
    action   : edge taken to reach this node (the neighbor name, same as state)
    g_cost   : actual travel time from start to this node  (g(n))
    h_cost   : heuristic estimate from this node to goal   (h(n))

    f_cost is computed as a property so it stays in sync automatically.
    """
    state   : str
    parent  : Optional["Node"]  = field(default=None, repr=False)
    action  : Optional[str]     = None
    g_cost  : float             = 0.0
    h_cost  : float             = 0.0

    @property
    def f_cost(self) -> float:
        """f(n) = g(n) + h(n)  — used by A*"""
        return self.g_cost + self.h_cost

    # Needed so Nodes can be compared in a priority queue
    def __lt__(self, other: "Node") -> bool:
        return self.f_cost < other.f_cost


# ---------------------------------------------------------------------------
# PATH RECONSTRUCTION  — shared by all algorithms
# ---------------------------------------------------------------------------
def reconstruct_path(node: Node) -> List[str]:
    """
    Walk up the parent chain from goal node back to start.
    Returns the path as a list of location names, start first.
    """
    path = []
    current = node
    while current is not None:
        path.append(current.state)
        current = current.parent
    path.reverse()
    return path


# ---------------------------------------------------------------------------
# GRAPH HELPER  — get neighbors cleanly
# ---------------------------------------------------------------------------
def get_neighbors(graph: dict, state: str) -> List[tuple]:
    """
    Return neighbors of a node as [(neighbor_name, travel_time), ...]
    Handles missing nodes gracefully.
    """
    return graph.get(state, [])


# ---------------------------------------------------------------------------
# RESULTS TABLE PRINTER  — used in test files
# ---------------------------------------------------------------------------
def print_results_table(results: List[SearchResult]):
    """
    Print a formatted comparison table for a list of SearchResult objects.
    All results should be for the same origin-destination pair.
    """
    if not results:
        return

    print(f"\n{'='*80}")
    print(f"  {results[0].start}  ->  {results[0].goal}")
    print(f"{'='*80}")
    print(f"  {'Algorithm':<10} {'Found':<6} {'Cost':>6}  {'Expanded':>9}  {'Max Front':>10}  Path")
    print(f"  {'-'*9} {'-'*5} {'-'*6}  {'-'*9}  {'-'*10}  {'-'*30}")

    for r in results:
        found_str = "YES" if r.found else "NO"
        cost_str  = f"{r.cost:.0f}" if r.found else "—"
        path_str  = r.path_str() if r.found else "—"
        # Truncate long paths for table display
        if len(path_str) > 45:
            path_str = path_str[:42] + "..."
        print(f"  {r.algorithm:<10} {found_str:<6} {cost_str:>6}  "
              f"{r.nodes_expanded:>9}  {r.max_frontier:>10}  {path_str}")

    print()


# ---------------------------------------------------------------------------
# SANITY CHECK
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Test the dataclasses and printer with dummy data
    r1 = SearchResult(
        algorithm="BFS", start="Saddar", goal="Gulshan",
        path=["Saddar", "Gulberg", "Liaquatabad", "Gulshan"],
        cost=49, nodes_expanded=8, max_frontier=5, found=True
    )
    r2 = SearchResult(
        algorithm="UCS", start="Saddar", goal="Gulshan",
        path=["Saddar", "Gulberg", "Liaquatabad", "Gulshan"],
        cost=49, nodes_expanded=12, max_frontier=7, found=True
    )
    r3 = SearchResult(
        algorithm="DFS", start="Saddar", goal="Gulshan",
        path=["Saddar", "Clifton", "DHA", "Korangi", "Johar", "Gulshan"],
        cost=123, nodes_expanded=15, max_frontier=4, found=True
    )

    print_results_table([r1, r2, r3])
    print("Node f_cost example:", Node(state="Saddar", g_cost=10, h_cost=5).f_cost)
    print("Path reconstruction:", reconstruct_path(
        Node("C", parent=Node("B", parent=Node("A")))
    ))
