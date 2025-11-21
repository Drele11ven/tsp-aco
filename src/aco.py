# src/aco.py
import numpy as np
import pandas as pd
from math import inf
from tqdm import trange
from .tsp_utils import make_run_folder, save_json
import os
from datetime import datetime

class AntColony:
    def __init__(self, dist_matrix, n_ants=20, n_iters=200,
                 alpha=1.0, beta=5.0, rho=0.5, q=1.0, seed=None):
        self.dist = np.array(dist_matrix)
        self.n = self.dist.shape[0]
        self.n_ants = n_ants
        self.n_iters = n_iters
        self.alpha = alpha
        self.beta = beta
        self.rho = rho  # evaporation rate
        self.q = q      # pheromone deposit factor
        rng = np.random.default_rng(seed)
        self.rng = rng
        # initial pheromone: small positive value
        self.pheromone = np.ones_like(self.dist) * 1.0
        np.fill_diagonal(self.pheromone, 0.0)
        # heuristic (1/d)
        with np.errstate(divide='ignore'):
            self.eta = 1.0 / (self.dist + 1e-12)
        np.fill_diagonal(self.eta, 0.0)
        # bookkeeping
        self.best_tour = None
        self.best_length = inf
        self.history = []  # per-iteration best length

    def _select_next(self, current, allowed):
        tau = self.pheromone[current, allowed] ** self.alpha
        eta = self.eta[current, allowed] ** self.beta
        probs = tau * eta
        s = probs.sum()
        if s == 0:
            # fallback: uniform
            return self.rng.choice(allowed)
        probs = probs / s
        return self.rng.choice(allowed, p=probs)

    def _tour_length(self, tour):
        # closed tour
        idx = np.array(tour)
        nxt = np.roll(idx, -1)
        return self.dist[idx, nxt].sum()

    def run(self, record_every=1, run_folder=None, verbose=False):
        if run_folder is None:
            run_folder = make_run_folder()
        params = dict(n_ants=self.n_ants, n_iters=self.n_iters,
                      alpha=self.alpha, beta=self.beta, rho=self.rho, q=self.q,
                      n_nodes=self.n, timestamp=datetime.now().isoformat())
        save_json(params, os.path.join(run_folder, "params.json"))

        per_iter_rows = []
        for it in range(self.n_iters):
            all_tours = []
            all_lengths = np.empty(self.n_ants)
            for a in range(self.n_ants):
                # construct a tour
                start = self.rng.integers(0, self.n)
                tour = [start]
                allowed = list(range(self.n))
                allowed.remove(start)
                current = start
                while allowed:
                    nxt = self._select_next(current, np.array(allowed))
                    tour.append(int(nxt))
                    allowed.remove(int(nxt))
                    current = int(nxt)
                # compute length
                L = self._tour_length(tour)
                all_tours.append(tour)
                all_lengths[a] = L

            # pheromone evaporation
            self.pheromone *= (1.0 - self.rho)
            # deposit
            for tour, L in zip(all_tours, all_lengths):
                deposit = self.q / (L + 1e-12)
                idx = np.array(tour)
                nxt = np.roll(idx, -1)
                self.pheromone[idx, nxt] += deposit
                self.pheromone[nxt, idx] += deposit

            # record best of iteration
            min_idx = np.argmin(all_lengths)
            iter_best_len = float(all_lengths[min_idx])
            iter_best_tour = all_tours[int(min_idx)]
            if iter_best_len < self.best_length:
                self.best_length = iter_best_len
                self.best_tour = iter_best_tour.copy()

            self.history.append(self.best_length)
            # log row
            per_iter_rows.append({
                "iter": it,
                "iter_best_length": float(iter_best_len),
                "global_best_length": float(self.best_length)
            })

            if verbose and (it % record_every == 0 or it == self.n_iters-1):
                print(f"iter {it}: iter_best={iter_best_len:.4f} global_best={self.best_length:.4f}")

        # save results
        df = pd.DataFrame(per_iter_rows)
        csv_path = os.path.join(run_folder, "per_iteration.csv")
        df.to_csv(csv_path, index=False)

        # save final summary
        summary = {
            "best_length": float(self.best_length),
            "best_tour": [int(i) for i in self.best_tour],
            "timestamp": datetime.now().isoformat()
        }
        save_json(summary, os.path.join(run_folder, "summary.json"))

        # save pheromone matrix numeric
        np.savetxt(os.path.join(run_folder, "pheromone.csv"), self.pheromone, delimiter=",")

        return {
            "run_folder": run_folder,
            "per_iteration_csv": csv_path,
            "summary": summary
        }
