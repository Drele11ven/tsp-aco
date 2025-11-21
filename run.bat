@echo off
REM ----------------------------
REM Setup and run TSP ACO Streamlit app
REM ----------------------------

REM Activate your virtual environment
call .venv\Scripts\activate

REM Install requirements (optional, first time)
pip install -r requirements.txt

REM Run Streamlit app
streamlit run streamlit_app.py

REM Pause so window stays open
pause
