# src/aco.py
import numpy as np
import pandas as pd
from math import inf
from .tsp_utils import make_run_folder, save_json
import os
from datetime import datetime

class AntColony:
    """
    Ant Colony implementation with an incremental step() method suitable for UI-driven iteration.
    Usage:
        ac = AntColony(dist_matrix, n_ants=..., n_iters=..., alpha=..., beta=..., rho=..., q=..., seed=...)
        for i in range(n_iters):
            stats = ac.step()   # performs one iteration, updates pheromones & history
        ac.save_run(run_folder)
    """
    def __init__(self, dist_matrix, n_ants=20, n_iters=200,
                 alpha=1.0, beta=5.0, rho=0.5, q=1.0, seed=None):
        self.dist = np.array(dist_matrix)
        self.n = self.dist.shape[0]
        self.n_ants = n_ants
        self.n_iters = n_iters
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.q = q
        self.rng = np.random.default_rng(seed)
        self.pheromone = np.ones_like(self.dist, dtype=float)
        np.fill_diagonal(self.pheromone, 0.0)
        with np.errstate(divide='ignore'):
            self.eta = 1.0 / (self.dist + 1e-12)
        np.fill_diagonal(self.eta, 0.0)

        # bookkeeping
        self.best_tour = None
        self.best_length = inf
        self.history = []            # global best length per iteration
        self._per_iter_rows = []     # internal store for saving per_iteration.csv
        self._iter_count = 0

    def _select_next(self, current, allowed):
        # allowed: numpy array of node indices
        tau = self.pheromone[current, allowed] ** self.alpha
        eta = self.eta[current, allowed] ** self.beta
        probs = tau * eta
        s = probs.sum()
        if s == 0:
            return int(self.rng.choice(allowed))
        probs = probs / s
        return int(self.rng.choice(allowed, p=probs))

    def _tour_length(self, tour):
        idx = np.array(tour)
        nxt = np.roll(idx, -1)
        return float(self.dist[idx, nxt].sum())

    def step(self):
        """
        Perform a single iteration (one round of self.n_ants constructing tours),
        update pheromones, update best tour/length, append internal history,
        and return stats dict for this iteration.
        """
        all_tours = []
        all_lengths = np.empty(self.n_ants, dtype=float)

        for a in range(self.n_ants):
            start = int(self.rng.integers(0, self.n))
            tour = [start]
            allowed = list(range(self.n))
            allowed.remove(start)
            current = start
            while allowed:
                nxt = self._select_next(current, np.array(allowed))
                tour.append(int(nxt))
                allowed.remove(int(nxt))
                current = int(nxt)
            L = self._tour_length(tour)
            all_tours.append(tour)
            all_lengths[a] = L

        # evaporation
        self.pheromone *= (1.0 - self.rho)

        # deposit
        for tour, L in zip(all_tours, all_lengths):
            deposit = self.q / (L + 1e-12)
            idx = np.array(tour, dtype=int)
            nxt = np.roll(idx, -1)
            self.pheromone[idx, nxt] += deposit
            self.pheromone[nxt, idx] += deposit

        # iteration-level best
        min_idx = int(np.argmin(all_lengths))
        iter_best_len = float(all_lengths[min_idx])
        iter_best_tour = all_tours[min_idx]

        if iter_best_len < self.best_length:
            self.best_length = iter_best_len
            self.best_tour = iter_best_tour.copy()

        self._iter_count += 1
        self.history.append(self.best_length)

        row = {
            "iter": self._iter_count - 1,
            "iter_best_length": float(iter_best_len),
            "global_best_length": float(self.best_length)
        }
        self._per_iter_rows.append(row)
        # return useful stats for UI
        return {
            "iter": row["iter"],
            "iter_best_length": row["iter_best_length"],
            "global_best_length": row["global_best_length"],
            "best_tour": (self.best_tour.copy() if self.best_tour is not None else None),
            "pheromone": self.pheromone.copy()
        }

    def save_run(self, run_folder=None, coords=None):
        """
        Save run artifacts into run_folder. If run_folder is None, create a timestamped folder.
        coords (optional): node coordinates to save as coords.csv
        Files saved:
          - params.json
          - per_iteration.csv
          - summary.json
          - pheromone.csv
          - coords.csv (if coords provided)
        """
        if run_folder is None:
            run_folder = make_run_folder()
        os.makedirs(run_folder, exist_ok=True)

        params = dict(
            n_ants=self.n_ants,
            n_iters=self._iter_count,
            alpha=self.alpha,
            beta=self.beta,
            rho=self.rho,
            q=self.q,
            n_nodes=self.n,
            timestamp=datetime.now().isoformat()
        )
        save_json(params, os.path.join(run_folder, "params.json"))

        # per-iteration CSV
        df = pd.DataFrame(self._per_iter_rows)
        df.to_csv(os.path.join(run_folder, "per_iteration.csv"), index=False)

        # summary
        summary = {
            "best_length": float(self.best_length) if self.best_length != inf else None,
            "best_tour": [int(i) for i in (self.best_tour or [])],
            "timestamp": datetime.now().isoformat()
        }
        save_json(summary, os.path.join(run_folder, "summary.json"))

        # pheromone matrix
        np.savetxt(os.path.join(run_folder, "pheromone.csv"), self.pheromone, delimiter=",")

        if coords is not None:
            try:
                pd.DataFrame(coords, columns=["x", "y"]).to_csv(os.path.join(run_folder, "coords.csv"), index=False)
            except Exception as e:
                print(f"Warning: could not save coords.csv: {e}")

        return run_folder

    def run(self, run_folder=None, coords=None, verbose=False):
        """
        Convenience: run until n_iters (self.n_iters) by calling step() repeatedly,
        then save using save_run().
        """
        target = int(self.n_iters)
        while self._iter_count < target:
            self.step()
        return self.save_run(run_folder=run_folder, coords=coords)
