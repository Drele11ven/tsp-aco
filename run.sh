#!/bin/bash
# ----------------------------
# Setup and run TSP ACO Streamlit app
# ----------------------------

# Activate virtual environment
source .venv/Scripts/activate

# Install requirements (first time only)
pip install -r requirements.txt

# Run Streamlit app
streamlit run streamlit_app.py
