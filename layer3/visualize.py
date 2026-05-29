"""
visualize_network.py
--------------------
Plots the best network configuration found by Layer 3
on a rough Karachi map outline using matplotlib.

Produces two charts:
  1. Best network — chosen stops and routes plotted on Karachi coordinates
  2. SA performance vs cooling rate — bar chart of avg scores
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from data.khi_graph import COORDINATES
from layer3.objective_func import objective, random_state, K_STOPS
from layer3.local_search import simulated_annealing

RANDOM_SEED = 42


def find_best_network(graph):
    """Run SA to find best network for visualisation."""
    random.seed(RANDOM_SEED)
    best_state = None
    best_score = -1
    for _ in range(20):
        result = simulated_annealing(
            graph, T_initial=10.0, cooling_rate=0.99,
            T_min=0.001, max_steps=8000
        )
        if result["score"] > best_score:
            best_score = result["score"]
            best_state = result["state"]
    return best_state, best_score


def plot_best_network(state, score, graph, save_path="best_network.png"):
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_facecolor("#f0f4f8")
    fig.patch.set_facecolor("#ffffff")

    active = set(state.active_stops)

    # Draw all candidate locations (inactive = grey dots)
    for loc, (lat, lon) in COORDINATES.items():
        if loc not in active:
            ax.scatter(lon, lat, s=40, color="#cccccc", zorder=2)
            ax.annotate(loc, (lon, lat), textcoords="offset points",
                        xytext=(4, 3), fontsize=6, color="#aaaaaa")

    # Draw routes (connections between active stops)
    for a, b in state.connections:
        if a in COORDINATES and b in COORDINATES:
            lat1, lon1 = COORDINATES[a]
            lat2, lon2 = COORDINATES[b]
            ax.plot([lon1, lon2], [lat1, lat2],
                    color="#2196F3", linewidth=2.0,
                    alpha=0.7, zorder=3)

    # Draw active stops
    for stop in state.active_stops:
        if stop in COORDINATES:
            lat, lon = COORDINATES[stop]
            is_hub = stop in ["Saddar", "Gulshan", "Orangi",
                               "DHA", "Korangi", "Surjani",
                               "North Nazimabad"]
            color  = "#e53935" if is_hub else "#43a047"
            size   = 180       if is_hub else 120
            ax.scatter(lon, lat, s=size, color=color,
                       edgecolors="white", linewidth=1.5, zorder=5)
            ax.annotate(stop, (lon, lat),
                        textcoords="offset points",
                        xytext=(5, 5), fontsize=8,
                        fontweight="bold", color="#1a1a2e")

    # Legend
    red_patch   = mpatches.Patch(color="#e53935", label="Key hub stop")
    green_patch = mpatches.Patch(color="#43a047", label="Regular stop")
    grey_patch  = mpatches.Patch(color="#cccccc", label="Inactive location")
    line_patch  = mpatches.Patch(color="#2196F3", label="Bus route")
    ax.legend(handles=[red_patch, green_patch, grey_patch, line_patch],
              loc="lower left", fontsize=9, framealpha=0.9)

    ax.set_title(
        f"Karachi Best Bus Network — Score: {score:.2f}/100\n"
        f"{K_STOPS} stops, {len(state.connections)} routes",
        fontsize=13, fontweight="bold", pad=15
    )
    ax.set_xlabel("Longitude", fontsize=10)
    ax.set_ylabel("Latitude",  fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {save_path}")
    plt.close()


def plot_sa_comparison(save_path="sa_cooling_comparison.png"):
    """Bar chart: SA average score vs cooling rate."""
    import random
    from data.khi_graph import build_graph
    graph = build_graph()

    rates   = [0.90, 0.95, 0.99]
    avgs    = []
    bests   = []
    worsts  = []

    for rate in rates:
        random.seed(RANDOM_SEED)
        scores = []
        for _ in range(50):
            r = simulated_annealing(
                graph, T_initial=10.0, cooling_rate=rate,
                T_min=0.01, max_steps=5000
            )
            scores.append(r["score"])
        avgs.append(sum(scores) / len(scores))
        bests.append(max(scores))
        worsts.append(min(scores))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_facecolor("#f9f9f9")

    x      = range(len(rates))
    width  = 0.25
    bars_a = ax.bar([i - width for i in x], avgs,   width, label="Avg",   color="#42a5f5")
    bars_b = ax.bar([i          for i in x], bests,  width, label="Best",  color="#66bb6a")
    bars_c = ax.bar([i + width for i in x], worsts, width, label="Worst", color="#ef5350")

    ax.set_xticks(list(x))
    ax.set_xticklabels([f"r = {r}" for r in rates], fontsize=11)
    ax.set_ylabel("Objective Score", fontsize=11)
    ax.set_title("SA Performance vs Cooling Rate (50 runs each)", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.set_ylim(30, 56)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    for bars in [bars_a, bars_b, bars_c]:
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.2,
                    f"{bar.get_height():.1f}",
                    ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {save_path}")
    plt.close()


if __name__ == "__main__":
    from data.khi_graph import build_graph
    graph = build_graph()

    print("Finding best network...")
    state, score = find_best_network(graph)

    print("Plotting best network map...")
    plot_best_network(state, score, graph,
                      save_path="/outputs/best_network.png")

    print("Plotting SA cooling rate comparison...")
    plot_sa_comparison(save_path="/outputs/sa_cooling_comparison.png")

    print("\nDone.")
