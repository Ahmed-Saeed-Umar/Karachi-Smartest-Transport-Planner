"""
karachi_graph.py
----------------
The core graph data for Karachi's transport network.

Nodes  : 22 real Karachi locations
Edges  : 42 bidirectional connections (84 directed)
Weights: estimated travel time in MINUTES under normal traffic

Bottleneck areas (poor connectivity, high travel time):
  - Orangi <-> Saddar  : must go through North Karachi or Liaquatabad
  - DHA    <-> Korangi : must cross the creek (long detour via Landhi)
  - Lyari  <-> SITE    : narrow roads, heavy congestion

Coordinate system: approximate lat/lng from Google Maps
  Used later by Layer 2 heuristics (h1 geographic distance).
"""

# ---------------------------------------------------------------------------
# NODE COORDINATES  {name: (latitude, longitude)}
# ---------------------------------------------------------------------------
COORDINATES = {
    "Saddar":           (24.8607,  67.0105),
    "Clifton":          (24.8138,  67.0300),
    "DHA":              (24.7920,  67.0721),
    "Korangi":          (24.8345,  67.1274),
    "Landhi":           (24.8567,  67.1490),
    "Malir":            (24.8934,  67.2014),
    "Gulshan":          (24.9215,  67.0956),
    "Johar":            (24.9042,  67.1341),
    "North Nazimabad":  (24.9430,  67.0356),
    "Nazimabad":        (24.9101,  67.0259),
    "Liaquatabad":      (24.9003,  67.0481),
    "North Karachi":    (24.9872,  67.0613),
    "Orangi":           (24.9620,  66.9988),
    "SITE":             (24.9282,  66.9923),
    "Lyari":            (24.8916,  66.9942),
    "Kemari":           (24.8399,  66.9750),
    "Garden":           (24.8740,  67.0230),
    "Gulberg":          (24.9010,  67.0653),
    "Buffer Zone":      (24.9651,  67.0822),
    "Surjani":          (25.0102,  67.0381),
    "Malir Cantt":      (24.9122,  67.2101),
    "Defence View":     (24.8321,  67.0561),
}

# ---------------------------------------------------------------------------
# EDGES  [(node_a, node_b, travel_time_minutes)]
# Bidirectional — the graph loader will add both directions.
#
# Weight rationale:
#   - Adjacent areas in the same zone: 8-15 min
#   - Cross-zone, good roads (Shahrae Faisal, M9): 15-25 min
#   - Bottleneck crossings (Creek, Lyari Nullah, Orangi hills): 30-50 min
# ---------------------------------------------------------------------------
EDGES = [
    # --- Saddar hub (central, connects south, north, east, west) ---
    ("Saddar",          "Clifton",          12),
    ("Saddar",          "Garden",           10),
    ("Saddar",          "Lyari",            18),
    ("Saddar",          "Nazimabad",        20),
    ("Saddar",          "Gulberg",          15),
    ("Saddar",          "Defence View",     20),

    # --- South / coastal ---
    ("Clifton",         "DHA",              18),
    ("Clifton",         "Defence View",     12),
    ("DHA",             "Defence View",     10),
    ("DHA",             "Korangi",          45),   # BOTTLENECK: creek crossing
    ("Kemari",          "Lyari",            20),
    ("Kemari",          "Saddar",           30),

    # --- East corridor ---
    ("Korangi",         "Landhi",           15),
    ("Korangi",         "Johar",            22),
    ("Landhi",          "Malir",            18),
    ("Landhi",          "Malir Cantt",      25),
    ("Malir",           "Malir Cantt",      14),
    ("Malir",           "Gulshan",          28),
    ("Johar",           "Gulshan",          16),
    ("Johar",           "Malir Cantt",      20),

    # --- Central / north-central ---
    ("Gulshan",         "Liaquatabad",      22),
    ("Gulshan",         "Buffer Zone",      18),
    ("Gulshan",         "North Nazimabad",  20),
    ("Gulberg",         "Liaquatabad",      12),
    ("Gulberg",         "Nazimabad",        10),
    ("Nazimabad",       "Liaquatabad",      12),
    ("Nazimabad",       "North Nazimabad",  15),
    ("Liaquatabad",     "North Nazimabad",  18),
    ("Liaquatabad",     "SITE",             22),

    # --- North ---
    ("North Nazimabad", "North Karachi",    20),
    ("North Nazimabad", "Orangi",           28),
    ("North Karachi",   "Surjani",          18),
    ("North Karachi",   "Buffer Zone",      16),
    ("Buffer Zone",     "Surjani",          22),
    ("Orangi",          "Surjani",          25),
    ("Orangi",          "SITE",             20),

    # --- West / industrial ---
    ("SITE",            "Lyari",            25),   # BOTTLENECK: narrow roads
    ("SITE",            "Garden",           28),
    ("Lyari",           "Garden",           15),
    ("Garden",          "Gulberg",          12),

    # --- Long cross-city ---
    ("Surjani",         "Malir Cantt",      55),   # very long cross-city
]

# ---------------------------------------------------------------------------
# GRAPH BUILDER
# Returns an adjacency list: {node: [(neighbor, weight), ...]}
# ---------------------------------------------------------------------------
def build_graph():
    """
    Build and return the Karachi transport graph as an adjacency list.

    Returns:
        dict: {location_name: [(neighbor_name, travel_time_minutes), ...]}
    """
    graph = {node: [] for node in COORDINATES}

    for node_a, node_b, weight in EDGES:
        if node_b not in [n for n, _ in graph[node_a]]:
            graph[node_a].append((node_b, weight))
        if node_a not in [n for n, _ in graph[node_b]]:
            graph[node_b].append((node_a, weight))

    return graph


# ---------------------------------------------------------------------------
# QUICK SANITY CHECK
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    graph = build_graph()

    total_nodes = len(graph)
    total_edges = sum(len(neighbors) for neighbors in graph.values()) // 2

    print(f"Nodes : {total_nodes}")
    print(f"Edges : {total_edges}")
    print()

    print("Low-connectivity nodes (<=2 connections):")
    for node, neighbors in graph.items():
        if len(neighbors) <= 2:
            print(f"  {node}: {[n for n, _ in neighbors]}")

    print()
    print("Neighbors of Saddar:")
    for neighbor, weight in sorted(graph["Saddar"], key=lambda x: x[1]):
        print(f"  -> {neighbor:20s}  {weight} min")
