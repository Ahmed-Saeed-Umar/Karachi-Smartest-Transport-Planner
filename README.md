# 🚌 Karachi's Smartest Transport Planner

An AI-powered transport planning system for Karachi built in two parts:
- **Route Finder** — finds optimal routes between any two locations using uninformed and heuristic search
- **Network Designer** — optimises bus stop placement and route connections using local search

---

## 📁 Project Structure

```
karachi_transport_planner/
│
├── README.md
├── requirements.txt
│
├── data/
│   ├── karachi_graph.py        # 22 nodes, 41 edges, lat/lng coordinates
│   └── karachi_graph.json      # Exportable graph data (for submission)
│
├── layer1/                     # Uninformed Search
│   ├── search_base.py          # Shared interface: SearchResult, Node, helpers
│   ├── uninformed_search.py    # BFS, DFS (cycle detection), UCS
│   └── test_layer1.py          # 5 test cases + results table + analysis
│
├── layer2/                     # Heuristic Search
│   ├── heuristics.py           # h1 (geographic), h2 (Karachi-aware) + consistency check
│   ├── informed_search.py      # Greedy Best-First, A*, IDA*
│   └── test_layer2.py          # Full comparison across all algorithms
│
└── layer3/                     # Network Optimisation
    ├── objective_function.py   # NetworkState, objective function, neighbourhood
    ├── local_search.py         # Hill Climbing, Random-Restart HC, Simulated Annealing
    ├── experiments.py          # 50-run experiments, 3 cooling schedules
    └── visualize_network.py    # Matplotlib plots of best network + SA comparison
```

---

## ⚙️ Setup

```bash
pip install matplotlib numpy
```

No other dependencies — the graph, search algorithms, and optimisation are all pure Python.

---

## 🚀 How to Run

### Layer 1 — Uninformed Search (BFS, DFS, UCS)

```bash
python layer1/test_layer1.py
```

Runs all three algorithms on 5 test cases and prints:
- Full results table (path, cost, nodes expanded, max frontier)
- Analysis comparing algorithm behaviour

### Layer 2 — Heuristic Search (Greedy, A\*, IDA\*)

```bash
python layer2/test_layer2.py
```

Runs 6 algorithms (UCS, Greedy-h1, Greedy-h2, A\*-h1, A\*-h2, IDA\*) on the same 5 test cases and prints:
- Nodes expanded comparison table
- A\* vs UCS improvement percentages
- Greedy suboptimality examples
- IDA\* memory vs computation trade-off

To verify heuristic admissibility and consistency:

```bash
python layer2/heuristics.py
```

### Layer 3 — Network Optimisation (Hill Climbing, Simulated Annealing)

```bash
python layer3/experiments.py
```

Runs the full experiment suite:
- Hill Climbing: 50 random starts, with and without sideways moves
- Simulated Annealing: cooling rates 0.90, 0.95, 0.99 × 50 runs each
- Finds and prints the best network configuration

To generate visualisation plots:

```bash
python layer3/visualize_network.py
```

Saves two images:
- `best_network.png` — chosen stops and routes on Karachi coordinates
- `sa_cooling_comparison.png` — bar chart of SA performance vs cooling rate

---

## 🗺️ The Graph

The Karachi transport network is modelled as an **undirected weighted graph**:

| Property | Value |
|---|---|
| Nodes | 22 real Karachi locations |
| Edges | 41 bidirectional connections |
| Weights | Estimated travel time in minutes |
| Coordinate system | Approximate lat/lng (Google Maps) |

**Key locations:** Saddar, Clifton, DHA, Korangi, Landhi, Malir, Gulshan, Johar, North Nazimabad, Nazimabad, Liaquatabad, North Karachi, Orangi, SITE, Lyari, Kemari, Garden, Gulberg, Buffer Zone, Surjani, Malir Cantt, Defence View

**Bottleneck areas:**
- `DHA → Korangi` — 45 min (Karachi Creek crossing, no direct bridge)
- `SITE → Lyari` — 25 min (narrow industrial roads, heavy congestion)
- `Orangi → Saddar` — indirect only (must route via SITE or North Nazimabad)

---

## 🧠 Algorithms

### Layer 1 — Uninformed Search

| Algorithm | Complete | Optimal | Notes |
|---|---|---|---|
| BFS | ✅ | ❌ (hops only) | Finds fewest stops, ignores travel time |
| DFS | ✅ (with cycle detection) | ❌ | Often finds very poor paths |
| UCS | ✅ | ✅ | Always finds minimum travel time |

### Layer 2 — Heuristic Search

| Algorithm | Complete | Optimal | Notes |
|---|---|---|---|
| Greedy (h1/h2) | ✅ | ❌ | Fast, but can find suboptimal paths |
| A\* (h1) | ✅ | ✅ | Geographic heuristic |
| A\* (h2) | ✅ | ✅ | Karachi-aware heuristic, expands fewer nodes |
| IDA\* | ✅ | ✅ | O(d) memory, more re-expansion |

**Heuristics:**
- **h1** — Haversine straight-line distance ÷ 25 km/h → minutes. Always admissible.
- **h2** — h1 + north-south penalty (×1.15) + congestion zone penalty (+1 min) + nullah crossing penalty (+8 min). Dominates h1.

### Layer 3 — Local Search

| Algorithm | Finds Global Optimum | Notes |
|---|---|---|
| Hill Climbing | ❌ | Gets stuck in ~2 steps on average |
| Random-Restart HC | Sometimes | 88–90% success rate across 50 starts |
| Simulated Annealing (r=0.99) | Best in practice | Best avg score: 49.44/100 |

**Objective function:** `0.40 × coverage + 0.35 × connectivity − 0.25 × efficiency_penalty`

---

## 📊 Key Results

**Layer 1 highlight** — Test: Orangi → DHA (bottleneck route)
- BFS: 131 min via Korangi (fewest hops)
- UCS: 88 min via SITE → Saddar → Clifton (optimal time) — **43 min faster**

**Layer 2 highlight** — A\*-h2 vs UCS nodes expanded:
- Test 2 (Surjani → Korangi): UCS expanded 27 nodes, A\*-h2 expanded 6 — **78% reduction**

**Layer 3 highlight** — SA (r=0.99) avg score 49.44 vs Hill Climbing avg 45.02 — **10% better**

