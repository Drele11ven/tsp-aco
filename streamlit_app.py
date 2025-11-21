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

st.set_page_config(page_title="TSP ACO Visualizer", layout="wide")

# ---------------- Sidebar: Experiment + Node source ----------------
st.sidebar.header("Experiment settings")

# Node source selector
st.sidebar.subheader("Node Source")
node_source = st.sidebar.selectbox("Select how to provide TSP nodes:",
                                   ["Random", "Fixed Seed", "Manual Input", "File Upload"])

n_nodes = st.sidebar.number_input("Number of nodes", 5, 200, 20, step=1)
seed_input = st.sidebar.number_input("Seed (0 = None)", 0, 999999, 0)

manual_data = st.sidebar.text_area(
    "Manual coordinates (one x,y per line)",
    value="",
    placeholder="10,20\n30,40\n50,10"
)

uploaded_file = st.sidebar.file_uploader("Upload CSV (columns: x,y)", type=["csv"])

# ACO parameters
st.sidebar.markdown("---")
n_ants = st.sidebar.slider("Number of ants", 1, 200, 40)
n_iters = st.sidebar.slider("Iterations", 1, 1000, 200)
alpha = st.sidebar.number_input("alpha (pheromone importance)", value=1.0, format="%.2f")
beta = st.sidebar.number_input("beta (heuristic importance)", value=5.0, format="%.2f")
rho = st.sidebar.number_input("rho (evaporation rate)", min_value=0.0, max_value=1.0, value=0.5, format="%.3f")
q = st.sidebar.number_input("q (deposit factor)", value=1.0, format="%.2f")

st.sidebar.markdown("---")
if st.sidebar.button("Load nodes"):
    # load depending on source
    def load_nodes():
        if node_source == "Random":
            seed = None if seed_input == 0 else int(seed_input)
            return random_euclidean_instance(int(n_nodes), seed=seed)
        if node_source == "Fixed Seed":
            return random_euclidean_instance(int(n_nodes), seed=12345)
        if node_source == "Manual Input":
            txt = manual_data.strip()
            if txt == "":
                st.warning("Please paste coordinates in the manual input box.")
                st.stop()
            rows = []
            for line in txt.splitlines():
                try:
                    x, y = map(float, line.split(","))
                    rows.append([x, y])
                except Exception:
                    st.error(f"Invalid line: {line}")
                    st.stop()
            return np.array(rows)
        if node_source == "File Upload":
            if uploaded_file is None:
                st.warning("Upload a CSV file first.")
                st.stop()
            df = pd.read_csv(uploaded_file)
            if not {"x", "y"}.issubset(df.columns):
                st.error("CSV must contain columns: x,y")
                st.stop()
            return df[["x", "y"]].values
        st.error("Unknown node source")
        st.stop()
    st.session_state["coords"] = load_nodes()

# If no coords in session, create default random instance
if "coords" not in st.session_state:
    st.session_state["coords"] = random_euclidean_instance(int(n_nodes), seed=(None if seed_input == 0 else int(seed_input)))

coords = np.array(st.session_state["coords"])
st.sidebar.write(f"Nodes: {coords.shape[0]}")

# ---------------- Run / Aggregation controls ----------------
col1, col2 = st.columns([2,1])
with col2:
    run_button = st.button("Run ACO")
    st.markdown("### Results")
    ensure_results_dir("results")
    if st.button("Show aggregated plot of all runs"):
        csvs = glob.glob("results/*/per_iteration.csv")
        if not csvs:
            st.info("No runs found in results/")
        else:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(10,5))
            for p in sorted(csvs):
                try:
                    df = pd.read_csv(p)
                    run_label = os.path.basename(os.path.dirname(p))
                    ax.plot(df['iter'], df['global_best_length'], label=run_label)
                except Exception:
                    continue
            ax.set_xlabel("Iteration")
            ax.set_ylabel("Global best length")
            ax.legend(fontsize=8)
            st.pyplot(fig)

with col1:
    st.header("TSP - Ant Colony Optimization")
    st.write("Interactive visualizer of ants and pheromones.")

# ---------------- Display node scatter ----------------
import matplotlib.pyplot as plt
col_small, _ = st.columns([3, 4])  # 20% width column

with col_small:
    fig_coords, axc = plt.subplots(figsize=(3, 3))
    axc.scatter(coords[:,0], coords[:,1], s=25)
    for i, (x, y) in enumerate(coords):
        axc.text(x, y, str(i), fontsize=6)
    axc.set_aspect('equal', adjustable='box')
    st.pyplot(fig_coords)

# ---------------- Run ACO (UI loop using step()) ----------------
if run_button:
    run_folder = make_run_folder("results")
    st.write(f"Running -> {run_folder}")

    # prepare distance matrix
    from src.tsp_utils import euclidean_distance_matrix
    D = euclidean_distance_matrix(coords)

    ac = AntColony(D, n_ants=int(n_ants), n_iters=int(n_iters),
                   alpha=float(alpha), beta=float(beta),
                   rho=float(rho), q=float(q),
                   seed=(None if int(seed_input) == 0 else int(seed_input)))

    placeholder = st.empty()
    progress_bar = st.progress(0)

    # iterate using step()
    for it in range(int(n_iters)):
        stats = ac.step()
        best = stats["best_tour"] or []
        # quick plot of best tour & pheromone
        fig1, ax1 = plt.subplots(figsize=(5,5))
        if best:
            visualizer.plot_tour(coords, best, ax=ax1, title=f"Best tour (len={stats['global_best_length']:.3f})")
        else:
            ax1.scatter(coords[:,0], coords[:,1])
            ax1.set_title("No tour yet")
        fig2, ax2 = plt.subplots(figsize=(5,5))
        visualizer.plot_pheromone_heat(coords, ac.pheromone, ax=ax2, title="Pheromone (relative)")

        with placeholder.container():
            st.write(f"Iteration {it+1} / {n_iters}")
            c1, c2 = st.columns(2)
            with c1:
                st.pyplot(fig1)
            with c2:
                st.pyplot(fig2)

        progress_bar.progress(min(100, int(100 * (it+1) / int(n_iters))))
        # small delay for visualization
        time.sleep(0.02)

    # After finishing iterations, save all run artifacts once
    ac.save_run(run_folder=run_folder, coords=coords)
    st.success(f"Run complete. Artifacts in: {run_folder}")
    st.write("Files in run folder:")
    st.write(sorted(os.listdir(run_folder)))
