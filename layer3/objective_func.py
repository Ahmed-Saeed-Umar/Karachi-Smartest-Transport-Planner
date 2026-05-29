"""
objective_function.py
---------------------
Defines the state representation and objective function for
the network design optimisation problem (Layer 3).

The Problem
-----------
KMC wants to select K bus stops from N candidate locations and
connect them with routes. Goal: maximise coverage and minimise
average travel time across the city.

State Representation
--------------------
A NetworkState is a complete candidate solution:
  - active_stops : set of K location names that are bus stops
  - connections  : set of (stop_a, stop_b) pairs representing routes

Objective Function
------------------
Score = w_coverage * coverage_score
      + w_connectivity * connectivity_score
      - w_efficiency * avg_travel_time_penalty

Weights chosen: coverage=0.4, connectivity=0.35, efficiency=0.25
Rationale:
  Coverage is weighted highest because Karachi's primary problem is
  that large areas (Orangi, Malir, Surjani) are simply unreachable.
  Connectivity is second — passengers need to be able to cross the city.
  Efficiency is third — travel time matters but only once reachability
  is satisfied.

Score is normalised to roughly [0, 100].
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import random
from itertools import combinations
from data.khi_graph import build_graph, COORDINATES


# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------
ALL_LOCATIONS  = list(COORDINATES.keys())   # 22 candidate locations
K_STOPS        = 10                          # number of bus stops to select
MAX_ROUTES     = 12                          # max connections between stops

# Objective function weights (must sum to 1.0)
W_COVERAGE     = 0.40
W_CONNECTIVITY = 0.35
W_EFFICIENCY   = 0.25

# Key hubs — connectivity between these is most important
KEY_HUBS = [
    "Saddar", "Gulshan", "North Nazimabad",
    "Korangi", "Orangi", "DHA", "Surjani"
]


# ---------------------------------------------------------------------------
# NETWORK STATE
# ---------------------------------------------------------------------------
class NetworkState:
    """
    Represents a complete candidate bus network configuration.

    Attributes
    ----------
    active_stops  : list of K location names selected as bus stops
    connections   : list of (stop_a, stop_b) tuples — the routes
    """

    def __init__(self, active_stops: list, connections: list):
        self.active_stops = list(active_stops)
        self.connections  = list(connections)

    def copy(self):
        return NetworkState(
            list(self.active_stops),
            list(self.connections)
        )

    def __repr__(self):
        return (f"NetworkState(stops={len(self.active_stops)}, "
                f"connections={len(self.connections)})")


# ---------------------------------------------------------------------------
# RANDOM INITIAL STATE
# ---------------------------------------------------------------------------
def random_state(graph: dict) -> NetworkState:
    """
    Generate a random valid NetworkState.
    Selects K random stops, then adds random connections between them.
    """
    stops = random.sample(ALL_LOCATIONS, K_STOPS)

    # Add random connections — only between active stops
    possible = list(combinations(stops, 2))
    n_connections = random.randint(K_STOPS - 1, min(MAX_ROUTES, len(possible)))
    connections = random.sample(possible, n_connections)

    return NetworkState(stops, connections)


# ---------------------------------------------------------------------------
# OBJECTIVE FUNCTION
# ---------------------------------------------------------------------------
def objective(state: NetworkState, graph: dict) -> float:
    """
    Evaluate a NetworkState. Higher score = better network.
    Returns a score in roughly [0, 100].

    Components
    ----------
    1. coverage_score     : what fraction of all 22 locations are active stops?
                            Bonus for covering geographically spread areas.

    2. connectivity_score : can passengers travel between all KEY_HUBS?
                            Uses BFS on the connection graph to check reachability.
                            Penalises isolated stops (no connections).

    3. efficiency_penalty : average number of transfers needed to get between
                            any two active stops. Fewer transfers = better.
    """

    stops       = set(state.active_stops)
    connections = set(map(frozenset, state.connections))

    # ------------------------------------------------------------------
    # 1. COVERAGE SCORE  [0, 1]
    # ------------------------------------------------------------------
    # Base: fraction of all locations covered
    base_coverage = len(stops) / len(ALL_LOCATIONS)

    # Bonus: are geographically distant zones represented?
    # Divide city into 4 zones by lat/lng quadrant
    zones = {"NW": 0, "NE": 0, "SW": 0, "SE": 0}
    lat_mid = 24.92   # approximate city centre latitude
    lon_mid = 67.05   # approximate city centre longitude

    for stop in stops:
        lat, lon = COORDINATES[stop]
        z = ("N" if lat >= lat_mid else "S") + ("E" if lon >= lon_mid else "W")
        zones[z] += 1

    zone_coverage = sum(1 for v in zones.values() if v > 0) / 4.0
    coverage_score = 0.6 * base_coverage + 0.4 * zone_coverage

    # ------------------------------------------------------------------
    # 2. CONNECTIVITY SCORE  [0, 1]
    # ------------------------------------------------------------------
    # Build adjacency list from connections
    adj = {s: set() for s in stops}
    for a, b in state.connections:
        if a in stops and b in stops:
            adj[a].add(b)
            adj[b].add(a)

    # BFS to find all reachable stops from the first active stop
    if stops:
        start    = next(iter(stops))
        visited  = {start}
        queue    = [start]
        while queue:
            node  = queue.pop(0)
            for nb in adj[node]:
                if nb not in visited:
                    visited.add(nb)
                    queue.append(nb)
        reachability = len(visited) / len(stops)
    else:
        reachability = 0.0

    # Hub coverage: how many KEY_HUBS are active stops?
    active_hubs  = sum(1 for h in KEY_HUBS if h in stops)
    hub_score    = active_hubs / len(KEY_HUBS)

    # Penalise isolated stops (stops with no connections)
    isolated     = sum(1 for s in stops if len(adj[s]) == 0)
    isolation_penalty = isolated / len(stops) if stops else 0

    connectivity_score = (
        0.5 * reachability
        + 0.35 * hub_score
        - 0.15 * isolation_penalty
    )
    connectivity_score = max(0.0, connectivity_score)

    # ------------------------------------------------------------------
    # 3. EFFICIENCY PENALTY  [0, 1]  (lower = better)
    # ------------------------------------------------------------------
    # BFS hop count between all pairs of connected stops
    # Normalise by worst-case (K_STOPS - 1 hops)
    total_hops = 0
    pair_count = 0

    stop_list = list(stops)
    for i in range(len(stop_list)):
        for j in range(i + 1, len(stop_list)):
            hops = _bfs_hops(adj, stop_list[i], stop_list[j])
            if hops is not None:
                total_hops += hops
                pair_count += 1

    if pair_count > 0:
        avg_hops = total_hops / pair_count
        efficiency_penalty = min(avg_hops / (K_STOPS - 1), 1.0)
    else:
        efficiency_penalty = 1.0   # worst case if disconnected

    # ------------------------------------------------------------------
    # FINAL SCORE
    # ------------------------------------------------------------------
    score = (
          W_COVERAGE     * coverage_score
        + W_CONNECTIVITY * connectivity_score
        - W_EFFICIENCY   * efficiency_penalty
    ) * 100   # scale to [0, 100]

    return round(score, 4)


def _bfs_hops(adj: dict, start: str, goal: str):
    """BFS hop count between two stops. Returns None if unreachable."""
    if start == goal:
        return 0
    visited = {start}
    queue   = [(start, 0)]
    while queue:
        node, hops = queue.pop(0)
        for nb in adj[node]:
            if nb == goal:
                return hops + 1
            if nb not in visited:
                visited.add(nb)
                queue.append((nb, hops + 1))
    return None


# ---------------------------------------------------------------------------
# NEIGHBOURHOOD FUNCTIONS
# ---------------------------------------------------------------------------
def get_neighbours(state: NetworkState, graph: dict) -> list:
    """
    Generate all neighbours of a NetworkState by applying one small change:

    Type 1 — Swap a stop:
        Remove one active stop, add one inactive stop.
        This changes which areas are covered.

    Type 2 — Add a connection:
        Add one new route between two active stops (if under MAX_ROUTES).

    Type 3 — Remove a connection:
        Remove one existing route (if more than K_STOPS-1 connections remain).

    Trade-off discussion:
        A sparse neighbourhood (only swaps) explores coverage changes
        but misses connectivity improvements. A dense neighbourhood
        (all three types) explores more but is slower per iteration.
        We use all three types for richer exploration.
    """
    neighbours = []
    stops      = set(state.active_stops)
    inactive   = [l for l in ALL_LOCATIONS if l not in stops]

    # Type 1: swap a stop
    for old_stop in state.active_stops:
        for new_stop in inactive:
            new_stops = [s if s != old_stop else new_stop
                         for s in state.active_stops]
            # Remove connections that used the old stop
            new_conns = [(a, b) for a, b in state.connections
                         if a != old_stop and b != old_stop]
            neighbours.append(NetworkState(new_stops, new_conns))

    # Type 2: add a connection
    if len(state.connections) < MAX_ROUTES:
        existing = set(map(frozenset, state.connections))
        for a, b in combinations(state.active_stops, 2):
            if frozenset({a, b}) not in existing:
                new_conns = state.connections + [(a, b)]
                neighbours.append(NetworkState(list(state.active_stops), new_conns))

    # Type 3: remove a connection
    if len(state.connections) > K_STOPS - 1:
        for i in range(len(state.connections)):
            new_conns = state.connections[:i] + state.connections[i+1:]
            neighbours.append(NetworkState(list(state.active_stops), new_conns))

    return neighbours


# ---------------------------------------------------------------------------
# SANITY CHECK
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    random.seed(42)
    graph = build_graph()

    print("Generating 5 random states and scoring them:\n")
    scores = []
    for i in range(5):
        state = random_state(graph)
        score = objective(state, graph)
        scores.append(score)
        print(f"  State {i+1}: stops={state.active_stops[:4]}... "
              f"connections={len(state.connections)}  score={score:.2f}")

    print(f"\nScore range: {min(scores):.2f} — {max(scores):.2f}")
    print(f"\nNeighbourhood size of a random state:")
    s = random_state(graph)
    nb = get_neighbours(s, graph)
    print(f"  {len(nb)} neighbours generated")
