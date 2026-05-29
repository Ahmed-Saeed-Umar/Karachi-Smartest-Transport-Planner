"""
test_layer2.py
--------------
Full comparison of all search algorithms on the same 5 test cases
from Layer 1. Includes UCS for baseline comparison.

Algorithms tested:
  UCS, Greedy(h1), Greedy(h2), A*(h1), A*(h2), IDA*(h2)

Report outputs:
  1. Full results table per test case
  2. Nodes expanded comparison (key metric)
  3. Whether Greedy ever found a non-optimal path
  4. A* improvement over UCS
  5. h2 vs h1 dominance check
  6. IDA* vs A* comparison
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from data.khi_graph import build_graph
from layer1.uninformed_search import ucs
from layer1.search_base import print_results_table
from layer2.informed_search import greedy, astar, idastar
from layer2.heuristics import h1, h2


# Same 5 test cases as Layer 1
TEST_CASES = [
    {"id": 1, "label": "Short direct route",                              "start": "Saddar",  "goal": "Clifton"},
    {"id": 2, "label": "Long cross-city route",                           "start": "Surjani", "goal": "Korangi"},
    {"id": 3, "label": "Route through bottleneck (Orangi -> DHA)",        "start": "Orangi",  "goal": "DHA"},
    {"id": 4, "label": "BFS vs UCS disagreement (Saddar -> Malir)",       "start": "Saddar",  "goal": "Malir"},
    {"id": 5, "label": "DFS worst case (Surjani -> Kemari)",              "start": "Surjani", "goal": "Kemari"},
]


def run_all(graph):
    all_results = {}

    for tc in TEST_CASES:
        s, g = tc["start"], tc["goal"]
        print(f"\nTest {tc['id']}: {tc['label']}")

        results = [
            ucs(graph, s, g),
            greedy(graph, s, g, h1),
            greedy(graph, s, g, h2),
            astar(graph,  s, g, h1),
            astar(graph,  s, g, h2),
            idastar(graph, s, g, h2),
        ]

        # Rename for clarity in table
        results[1].algorithm = "Greedy-h1"
        results[2].algorithm = "Greedy-h2"
        results[3].algorithm = "A*-h1"
        results[4].algorithm = "A*-h2"

        print_results_table(results)
        all_results[tc["id"]] = results

    return all_results


def print_analysis(all_results):
    print("\n" + "="*80)
    print("  LAYER 2 ANALYSIS")
    print("="*80)

    algo_names = ["UCS", "Greedy-h1", "Greedy-h2", "A*-h1", "A*-h2", "IDA*"]

    # 1. Nodes expanded table
    print(f"\n[1] Nodes Expanded Comparison:")
    print(f"  {'Test':<6} " + " ".join(f"{a:>11}" for a in algo_names))
    print("  " + "-"*72)
    for tc_id, results in all_results.items():
        row = f"  {tc_id:<6}"
        for name in algo_names:
            r = next((r for r in results if r.algorithm == name), None)
            row += f" {r.nodes_expanded:>11}" if r else f" {'N/A':>11}"
        print(row)

    # 2. A* improvement over UCS
    print(f"\n[2] A*-h2 nodes expanded vs UCS (reduction %):")
    for tc_id, results in all_results.items():
        ucs_r  = next(r for r in results if r.algorithm == "UCS")
        astar_r = next(r for r in results if r.algorithm == "A*-h2")
        if ucs_r.nodes_expanded > 0:
            reduction = (1 - astar_r.nodes_expanded / ucs_r.nodes_expanded) * 100
            print(f"  Test {tc_id}: UCS={ucs_r.nodes_expanded}, "
                  f"A*-h2={astar_r.nodes_expanded}, "
                  f"reduction={reduction:.1f}%")

    # 3. Does Greedy ever find non-optimal paths?
    print(f"\n[3] Greedy suboptimality check:")
    any_suboptimal = False
    for tc_id, results in all_results.items():
        ucs_r  = next(r for r in results if r.algorithm == "UCS")
        g_h1   = next(r for r in results if r.algorithm == "Greedy-h1")
        g_h2   = next(r for r in results if r.algorithm == "Greedy-h2")
        for g_r in [g_h1, g_h2]:
            if g_r.found and g_r.cost > ucs_r.cost:
                any_suboptimal = True
                print(f"  Test {tc_id} ({g_r.algorithm}): "
                      f"Greedy cost={g_r.cost:.0f}, Optimal={ucs_r.cost:.0f} "
                      f"-- SUBOPTIMAL by {g_r.cost - ucs_r.cost:.0f} min")
                print(f"    Greedy path : {g_r.path_str()}")
                print(f"    Optimal path: {ucs_r.path_str()}")
    if not any_suboptimal:
        print("  Greedy found optimal paths on all test cases.")

    # 4. h2 dominates h1?
    print(f"\n[4] h2 vs h1 dominance (nodes expanded):")
    print(f"  {'Test':<6} {'A*-h1':>8} {'A*-h2':>8} {'h2 better?':>12}")
    print(f"  {'-'*36}")
    for tc_id, results in all_results.items():
        h1_r = next(r for r in results if r.algorithm == "A*-h1")
        h2_r = next(r for r in results if r.algorithm == "A*-h2")
        better = "YES" if h2_r.nodes_expanded <= h1_r.nodes_expanded else "NO"
        print(f"  {tc_id:<6} {h1_r.nodes_expanded:>8} {h2_r.nodes_expanded:>8} {better:>12}")

    # 5. IDA* vs A* comparison
    print(f"\n[5] IDA* vs A*-h2 (memory vs computation trade-off):")
    print(f"  {'Test':<6} {'A* exp':>8} {'IDA* exp':>10} {'A* front':>10} {'IDA* front':>12}")
    print(f"  {'-'*48}")
    for tc_id, results in all_results.items():
        a_r   = next(r for r in results if r.algorithm == "A*-h2")
        ida_r = next(r for r in results if r.algorithm == "IDA*")
        print(f"  {tc_id:<6} {a_r.nodes_expanded:>8} {ida_r.nodes_expanded:>10} "
              f"{a_r.max_frontier:>10} {ida_r.max_frontier:>12}")

    print()


if __name__ == "__main__":
    graph       = build_graph()
    all_results = run_all(graph)
    print_analysis(all_results)
