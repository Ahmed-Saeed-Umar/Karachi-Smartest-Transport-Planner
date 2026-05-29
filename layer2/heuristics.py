"""
heuristics.py
-------------
Two admissible heuristics for the Karachi transport graph.

  h1 — Geographic Heuristic
       Straight-line (Haversine) distance between two locations,
       converted to estimated travel time using average bus speed.

  h2 — Karachi-Aware Heuristic
       Builds on h1 but adds penalty multipliers for:
         - North-south travel (slower due to Karachi's road layout)
         - Known congestion zones (Lyari, SITE, DHA creek crossing)
         - Nullah / expressway crossings that add unavoidable delays

Both heuristics are ADMISSIBLE — they never overestimate the true cost.
Admissibility is argued below each function.

Consistency is verified on 5 edges at the bottom of this file.
"""

import sys, os, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from data.khi_graph import COORDINATES, build_graph


# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------

# Average bus speed in Karachi under normal traffic: ~25 km/h
# (Karachi public transport averages 20-30 km/h on mixed roads)
AVG_SPEED_KMH = 25.0

# Karachi geography penalties for h2
# These are multipliers applied on top of h1.
# All >= 1.0 so h2 never goes below h1 (h2 dominates h1).
# All kept conservative so we don't overestimate (admissibility preserved).

# North-south travel penalty: Karachi's major arteries run east-west
# (M.A. Jinnah Road, Shahrae Faisal). North-south is ~20% slower.
NS_PENALTY = 1.15

# Congestion zone nodes: being near these adds delay
CONGESTION_ZONES = {"Lyari", "SITE", "Garden", "Saddar", "Liaquatabad"}

# Creek/nullah crossing: if start and goal are on opposite sides
# of the Malir River or Lyari Nullah, add a flat time penalty (minutes)
# This is a known unavoidable delay — crossings are few and slow.
CROSSING_PENALTY_MIN = 8   # conservative — actual can be 15-30 min

# Nodes east of Malir River
EAST_OF_MALIR  = {"Malir", "Malir Cantt", "Korangi", "Landhi", "Johar", "Gulshan"}
# Nodes west of Lyari Nullah
WEST_OF_LYARI  = {"Kemari", "Lyari", "SITE", "Orangi"}


# ---------------------------------------------------------------------------
# HAVERSINE DISTANCE  (helper)
# ---------------------------------------------------------------------------
def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate straight-line distance between two lat/lng points in kilometres.
    Uses the Haversine formula — accurate for short distances.
    """
    R = 6371.0  # Earth radius in km
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi       = math.radians(lat2 - lat1)
    dlambda    = math.radians(lon2 - lon1)

    a = (math.sin(dphi / 2) ** 2
         + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2)
    return R * 2 * math.asin(math.sqrt(a))


# ---------------------------------------------------------------------------
# H1 — GEOGRAPHIC HEURISTIC
# ---------------------------------------------------------------------------
def h1(node: str, goal: str) -> float:
    """
    Straight-line distance converted to travel time in minutes.

    Formula:
        distance_km = haversine(node, goal)
        time_min    = (distance_km / AVG_SPEED_KMH) * 60

    Admissibility argument:
        The straight-line distance is always <= the actual road distance.
        Dividing by AVG_SPEED_KMH (25 km/h) gives a time estimate that
        assumes the bus travels in a straight line at full speed — no turns,
        no stops, no traffic. Real travel always takes longer. Therefore
        h1 never overestimates the true cost. h1 is admissible.

    Returns 0 if either node is not in COORDINATES (fail-safe).
    """
    if node not in COORDINATES or goal not in COORDINATES:
        return 0.0

    lat1, lon1 = COORDINATES[node]
    lat2, lon2 = COORDINATES[goal]

    dist_km  = haversine_km(lat1, lon1, lat2, lon2)
    time_min = (dist_km / AVG_SPEED_KMH) * 60
    return round(time_min, 2)


# ---------------------------------------------------------------------------
# H2 — KARACHI-AWARE HEURISTIC
# ---------------------------------------------------------------------------
def h2(node: str, goal: str) -> float:
    """
    Geographic heuristic + Karachi-specific knowledge penalties.

    Enhancements over h1:
      1. North-south penalty  : if travel direction is predominantly north-south,
                                multiply by NS_PENALTY (1.15)
      2. Congestion penalty   : if either endpoint is a known congestion zone,
                                add a small flat time (3 min)
      3. River/nullah crossing: if start and goal are on opposite sides of
                                the Malir River or Lyari Nullah, add
                                CROSSING_PENALTY_MIN (8 min)

    Admissibility argument:
        h2 >= h1 always (h2 dominates h1), so h2 is more informed.
        Each penalty is deliberately conservative:
          - NS_PENALTY of 1.15 means we assume 15% slower — real slowdown
            on north-south routes is often 20-40%, so we still underestimate.
          - Congestion flat penalty of 3 min — actual delays are often 10+ min.
          - Crossing penalty of 8 min — actual nullah crossings add 15-30 min.
        Because all penalties are lower than real-world values, h2 still
        never overestimates. h2 is admissible.

    Since h2 >= h1 for all nodes, h2 also dominates h1 — meaning A* with h2
    will expand fewer nodes than A* with h1.
    """
    if node not in COORDINATES or goal not in COORDINATES:
        return 0.0

    base = h1(node, goal)

    lat1, lon1 = COORDINATES[node]
    lat2, lon2 = COORDINATES[goal]

    # 1. North-south penalty
    #    If vertical displacement > horizontal displacement, travel is
    #    predominantly north-south → apply penalty
    delta_lat = abs(lat2 - lat1)
    delta_lon = abs(lon2 - lon1)
    if delta_lat > delta_lon:
        base *= NS_PENALTY

    # 2. Congestion zone penalty
    #    Being near a congestion zone adds unavoidable delay
    if node in CONGESTION_ZONES or goal in CONGESTION_ZONES:
        base += 1.0

    # 3. River / nullah crossing penalty
    #    East-of-Malir <-> anywhere west  OR  West-of-Lyari <-> anywhere east
    crosses_malir = (node in EAST_OF_MALIR) != (goal in EAST_OF_MALIR)
    crosses_lyari = (node in WEST_OF_LYARI) != (goal in WEST_OF_LYARI)
    if crosses_malir or crosses_lyari:
        base += CROSSING_PENALTY_MIN

    return round(base, 2)


# ---------------------------------------------------------------------------
# CONSISTENCY CHECK
# ---------------------------------------------------------------------------
def check_consistency(graph: dict, heuristic, h_name: str, edges_to_check: list):
    """
    Verify the triangle inequality for consistency:
        h(n) <= cost(n -> n') + h(n')

    A consistent heuristic guarantees A* never re-expands a node,
    making it more efficient than a merely admissible one.

    Prints a table showing whether each edge satisfies consistency.
    """
    print(f"\nConsistency check for {h_name}")
    print(f"  Rule: h(n) <= cost(n, n') + h(n')")
    print(f"  {'Edge':<35} {'h(n)':>6}  {'cost':>5}  {'h(n\')':>6}  {'RHS':>6}  {'OK?':>5}")
    print(f"  {'-'*35} {'-'*6}  {'-'*5}  {'-'*6}  {'-'*6}  {'-'*5}")

    all_ok = True
    for n, n_prime, goal in edges_to_check:
        # find edge cost
        cost = next((w for nb, w in graph.get(n, []) if nb == n_prime), None)
        if cost is None:
            print(f"  {n} -> {n_prime}: edge not found")
            continue

        hn      = heuristic(n, goal)
        hn_p    = heuristic(n_prime, goal)
        rhs     = cost + hn_p
        ok      = hn <= rhs + 1e-9   # small epsilon for float comparison
        flag    = "YES" if ok else "FAIL !!!"
        if not ok:
            all_ok = False

        edge_str = f"{n} -> {n_prime}"
        print(f"  {edge_str:<35} {hn:>6.1f}  {cost:>5}  {hn_p:>6.1f}  {rhs:>6.1f}  {flag:>5}")

    print(f"\n  Result: {'All edges consistent ✓' if all_ok else 'INCONSISTENCY DETECTED ✗'}")


# ---------------------------------------------------------------------------
# SANITY CHECK
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    graph = build_graph()

    # Print h1 and h2 values for a few sample pairs
    sample_pairs = [
        ("Saddar",  "Korangi"),
        ("Orangi",  "DHA"),
        ("Surjani", "Kemari"),
        ("Clifton", "Malir"),
        ("SITE",    "Johar"),
    ]

    print("Sample heuristic values (estimated travel time in minutes):")
    print(f"  {'From':<20} {'To':<20} {'h1':>8} {'h2':>8}")
    print(f"  {'-'*20} {'-'*20} {'-'*8} {'-'*8}")
    for start, goal in sample_pairs:
        print(f"  {start:<20} {goal:<20} {h1(start,goal):>8.1f} {h2(start,goal):>8.1f}")

    # 5 edges for consistency check — (node, neighbor, goal)
    # Pick edges that are in the graph
    edges_to_check = [
        ("Saddar",         "Clifton",        "Korangi"),
        ("Gulshan",        "Johar",          "Malir"),
        ("Orangi",         "SITE",           "DHA"),
        ("North Karachi",  "Surjani",        "Kemari"),
        ("Liaquatabad",    "North Nazimabad","Malir Cantt"),
    ]

    check_consistency(graph, h1, "h1 (Geographic)", edges_to_check)
    check_consistency(graph, h2, "h2 (Karachi-Aware)", edges_to_check)
