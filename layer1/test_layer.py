"""
test_layer1.py
--------------
Runs all three uninformed search algorithms on 5 deliberately chosen
origin-destination pairs and prints the full comparison table.

Test case design rationale
--------------------------
1. Saddar      -> Clifton       SHORT direct route (1 hop, 12 min)
2. Surjani     -> Korangi       LONG cross-city (north to east, many hops)
3. Orangi      -> DHA           BOTTLENECK route (must cross city through poor links)
4. Saddar      -> Malir         BFS vs UCS DISAGREE (weights vary a lot on alternate paths)
5. Surjani     -> Kemari        DFS WORST CASE (deep dead-end branches on the way)
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from data.khi_graph import build_graph
from layer1.uninformed_search import bfs, dfs, ucs
from layer1.search_base import print_results_table, SearchResult
from typing import List


# ---------------------------------------------------------------------------
# TEST CASES
# ---------------------------------------------------------------------------
TEST_CASES = [
    {
        "id"     : 1,
        "label"  : "Short direct route",
        "start"  : "Saddar",
        "goal"   : "Clifton",
    },
    {
        "id"     : 2,
        "label"  : "Long cross-city route",
        "start"  : "Surjani",
        "goal"   : "Korangi",
    },
    {
        "id"     : 3,
        "label"  : "Route through bottleneck (Orangi -> DHA creek crossing)",
        "start"  : "Orangi",
        "goal"   : "DHA",
    },
    {
        "id"     : 4,
        "label"  : "BFS and UCS find different paths (varying edge weights)",
        "start"  : "Saddar",
        "goal"   : "Malir",
    },
    {
        "id"     : 5,
        "label"  : "DFS performs badly (deep dead ends)",
        "start"  : "Surjani",
        "goal"   : "Kemari",
    },
]


# ---------------------------------------------------------------------------
# RUN ALL TESTS
# ---------------------------------------------------------------------------
def run_all_tests(graph: dict) -> dict:
    """
    Run BFS, DFS, UCS on all 5 test cases.
    Returns a dict of results keyed by test case id.
    """
    all_results = {}
    algorithms  = [bfs, dfs, ucs]

    for tc in TEST_CASES:
        print(f"\nTest {tc['id']}: {tc['label']}")
        results = [algo(graph, tc['start'], tc['goal']) for algo in algorithms]
        print_results_table(results)

        # Print full paths separately (table truncates long paths)
        for r in results:
            print(f"  {r.algorithm:4s} path: {r.path_str()}")

        all_results[tc['id']] = results

    return all_results


# ---------------------------------------------------------------------------
# ANALYSIS SUMMARY
# ---------------------------------------------------------------------------
def print_analysis(all_results: dict):
    """
    Print a structured analysis comparing algorithm behaviour across test cases.
    This feeds directly into the report's 200-300 word analysis section.
    """
    print("\n" + "="*80)
    print("  ANALYSIS SUMMARY")
    print("="*80)

    # 1. Where did BFS and UCS disagree on cost?
    print("\n[1] BFS vs UCS — cost disagreements:")
    for tc_id, results in all_results.items():
        bfs_r = next(r for r in results if r.algorithm == "BFS")
        ucs_r = next(r for r in results if r.algorithm == "UCS")
        if bfs_r.found and ucs_r.found and bfs_r.cost != ucs_r.cost:
            print(f"  Test {tc_id}: BFS cost={bfs_r.cost:.0f} min  |  "
                  f"UCS cost={ucs_r.cost:.0f} min  "
                  f"(UCS saved {bfs_r.cost - ucs_r.cost:.0f} min)")
        elif bfs_r.found and ucs_r.found:
            print(f"  Test {tc_id}: BFS and UCS agreed — cost={bfs_r.cost:.0f} min")

    # 2. Where did DFS waste effort (much higher cost than UCS)?
    print("\n[2] DFS cost vs UCS optimal cost:")
    for tc_id, results in all_results.items():
        dfs_r = next(r for r in results if r.algorithm == "DFS")
        ucs_r = next(r for r in results if r.algorithm == "UCS")
        if dfs_r.found and ucs_r.found:
            overhead = dfs_r.cost - ucs_r.cost
            flag = "  *** SUBOPTIMAL ***" if overhead > 0 else ""
            print(f"  Test {tc_id}: DFS={dfs_r.cost:.0f} min, "
                  f"UCS={ucs_r.cost:.0f} min, "
                  f"overhead={overhead:.0f} min{flag}")

    # 3. Nodes expanded comparison
    print("\n[3] Nodes expanded (search effort):")
    header = f"  {'Test':<6} {'BFS':>8} {'DFS':>8} {'UCS':>8}"
    print(header)
    print("  " + "-"*30)
    for tc_id, results in all_results.items():
        row = f"  {tc_id:<6}"
        for algo_name in ["BFS", "DFS", "UCS"]:
            r = next(r for r in results if r.algorithm == algo_name)
            row += f" {r.nodes_expanded:>8}"
        print(row)

    # 4. Path length (number of stops) comparison
    print("\n[4] Path length (number of stops):")
    header = f"  {'Test':<6} {'BFS':>8} {'DFS':>8} {'UCS':>8}"
    print(header)
    print("  " + "-"*30)
    for tc_id, results in all_results.items():
        row = f"  {tc_id:<6}"
        for algo_name in ["BFS", "DFS", "UCS"]:
            r = next(r for r in results if r.algorithm == algo_name)
            stops = len(r.path) if r.found else 0
            row += f" {stops:>8}"
        print(row)

    print()


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    graph       = build_graph()
    all_results = run_all_tests(graph)
    print_analysis(all_results)
