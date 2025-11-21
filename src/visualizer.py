# src/visualizer.py
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

def plot_tour(coords, tour, ax=None, title=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(6,6))
    tour_idx = list(tour) + [tour[0]]
    xs = coords[tour_idx, 0]
    ys = coords[tour_idx, 1]
    ax.plot(xs, ys, "-", linewidth=1.5)
    ax.scatter(coords[:,0], coords[:,1], s=40)
    for i,(x,y) in enumerate(coords):
        ax.text(x, y, str(i), fontsize=9)
    if title:
        ax.set_title(title)
    ax.set_aspect('equal', adjustable='box')
    return ax

def plot_pheromone_heat(coords, pheromone, ax=None, title=None):
    # Build edges with pheromone weight and plot as lines with alpha proportional to pheromone
    if ax is None:
        fig, ax = plt.subplots(figsize=(6,6))
    n = coords.shape[0]
    max_p = np.max(pheromone)
    if max_p <= 0:
        max_p = 1.0
    for i in range(n):
        for j in range(i+1, n):
            w = pheromone[i,j] / max_p
            ax.plot([coords[i,0], coords[j,0]], [coords[i,1], coords[j,1]],
                    linewidth=2 * w + 0.1, alpha=0.6*w)
    ax.scatter(coords[:,0], coords[:,1], s=30)
    if title:
        ax.set_title(title)
    ax.set_aspect('equal', adjustable='box')
    return ax

def save_plot(fig, path, dpi=150):
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)

def plot_per_iteration(csv_path, savepath=None):
    df = pd.read_csv(csv_path)
    fig, ax = plt.subplots(figsize=(8,4))
    ax.plot(df['iter'], df['iter_best_length'], label='iter_best')
    ax.plot(df['iter'], df['global_best_length'], label='global_best')
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Tour length")
    ax.legend()
    if savepath:
        fig.savefig(savepath, dpi=150)
        plt.close(fig)
    return fig
