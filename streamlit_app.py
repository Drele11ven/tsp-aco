# streamlit_app.py
import streamlit as st
import numpy as np
import os
from src.tsp_utils import random_euclidean_instance, make_run_folder, ensure_results_dir
from src.aco import AntColony
from src import visualizer
from datetime import datetime
import time
import pandas as pd
import glob
import json

st.set_page_config(page_title="TSP ACO Visualizer", layout="wide")

st.sidebar.header("Experiment settings")
n_nodes = st.sidebar.slider("Number of nodes", 5, 60, 20)
n_ants = st.sidebar.slider("Number of ants", 1, 200, 40)
n_iters = st.sidebar.slider("Iterations", 1, 1000, 200)
alpha = st.sidebar.number_input("alpha (pheromone importance)", value=1.0, format="%.2f")
beta = st.sidebar.number_input("beta (heuristic importance)", value=5.0, format="%.2f")
rho = st.sidebar.number_input("rho (evaporation rate)", min_value=0.0, max_value=1.0, value=0.5, format="%.3f")
q = st.sidebar.number_input("q (deposit factor)", value=1.0, format="%.2f")
seed = st.sidebar.number_input("random seed (0 = random)", value=0)

st.sidebar.markdown("---")
if st.sidebar.button("Generate random instance"):
    coords = random_euclidean_instance(n_nodes, seed=(None if seed==0 else int(seed)))
    st.session_state['coords'] = coords.tolist()
else:
    if 'coords' not in st.session_state:
        st.session_state['coords'] = random_euclidean_instance(n_nodes, seed=(None if seed==0 else int(seed))).tolist()

coords = np.array(st.session_state['coords'])
st.sidebar.write(f"Nodes: {coords.shape[0]}")

# Run controls
col1, col2 = st.columns([2,1])
with col2:
    run_button = st.button("Run ACO")
    st.markdown("### Results")
    ensure_results_dir("results")
    if st.button("Show aggregated plot of all runs"):
        # aggregate
        csvs = glob.glob("results/*/per_iteration.csv")
        if not csvs:
            st.info("No runs found in results/")
        else:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(8,4))
            for p in csvs:
                df = pd.read_csv(p)
                run_label = os.path.basename(os.path.dirname(p))
                ax.plot(df['iter'], df['global_best_length'], label=run_label)
            ax.set_xlabel("Iteration")
            ax.set_ylabel("Global best length")
            ax.legend(fontsize=8)
            st.pyplot(fig)

with col1:
    st.header("TSP - Ant Colony Optimization")
    st.write("Interactive visualizer of ants and pheromones.")

# display coordinate scatter and index
import matplotlib.pyplot as plt
fig_coords, axc = plt.subplots(figsize=(5,5))
axc.scatter(coords[:,0], coords[:,1], s=40)
for i,(x,y) in enumerate(coords):
    axc.text(x, y, str(i))
axc.set_title("Node positions")
axc.set_aspect('equal', adjustable='box')
st.pyplot(fig_coords)

if run_button:
    run_folder = make_run_folder("results")
    st.write(f"Running -> {run_folder}")

    # prepare distance matrix
    from src.tsp_utils import euclidean_distance_matrix
    D = euclidean_distance_matrix(coords)

    ac = AntColony(D, n_ants=n_ants, n_iters=n_iters,
                   alpha=alpha, beta=beta, rho=rho, q=q,
                   seed=(None if seed==0 else int(seed)))

    placeholder = st.empty()
    progress_bar = st.progress(0)
    # Run and animate by saving per-iteration figures and showing them
    for it in range(ac.n_iters):
        # run single iteration by calling internal loop: to avoid large changes, run run() with n_iters=1 repeatedly is heavy.
        # Instead simulate by taking advantage of existing run() behaviour. For simplicity we call run() on a copy with 1 iteration.
        ac.n_iters = 1
        ac.n_ants = n_ants
        res = ac.run(record_every=1, run_folder=run_folder, verbose=False)
        # after run() returned, the 'history' appended data
        # make quick plots: current best tour & pheromone heat
        best = ac.best_tour
        fig1, ax1 = plt.subplots(figsize=(5,5))
        visualizer.plot_tour(coords, best, ax=ax1, title=f"Best tour (len={ac.best_length:.3f})")
        fig2, ax2 = plt.subplots(figsize=(5,5))
        visualizer.plot_pheromone_heat(coords, ac.pheromone, ax=ax2, title="Pheromone (relative)")

        # layout in placeholder
        with placeholder.container():
            st.write(f"Iteration {len(ac.history)} / {n_iters}")
            c1, c2 = st.columns(2)
            with c1:
                st.pyplot(fig1)
            with c2:
                st.pyplot(fig2)
        progress_bar.progress(min(100, int(100*len(ac.history)/n_iters)))
        # small delay for visualization
        time.sleep(0.05)
        # stop if reached desired iterations
        if len(ac.history) >= n_iters:
            break

    # Save final artifacts (per-iteration csv etc were saved by run()).
    # save final trajectory plot & per-iteration plot
    from src.tsp_utils import save_json
    import numpy as np
    # Save best tour figure
    fbest = os.path.join(run_folder, "best_tour.png")
    fig, ax = plt.subplots(figsize=(6,6))
    visualizer.plot_tour(coords, ac.best_tour, ax=ax, title=f"Best tour (len={ac.best_length:.3f})")
    fig.savefig(fbest, dpi=150)
    plt.close(fig)

    # Save pheromone heat
    fph = os.path.join(run_folder, "pheromone_heat.png")
    fig, ax = plt.subplots(figsize=(6,6))
    visualizer.plot_pheromone_heat(coords, ac.pheromone, ax=ax, title="Pheromone")
    fig.savefig(fph, dpi=150)
    plt.close(fig)

    # Save coords
    pd.DataFrame(coords, columns=["x","y"]).to_csv(os.path.join(run_folder, "coords.csv"), index=False)

    st.success(f"Run complete. Artifacts in: {run_folder}")
    st.write("Files in run folder:")
    st.write(os.listdir(run_folder))
