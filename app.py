import os
import subprocess
import sys

import streamlit as st

from football_predictions_view import (
    show_football_feedback,
    show_football_predictions,
    show_pool_draws,
)
from picks_view import show_basketball_picks
from results_view import show_results

DEVELOPER = "ENGR Maurizo"
APP_NAME = "Sports Predictor"

BASE = os.path.dirname(os.path.abspath(__file__))

st.set_page_config(page_title=APP_NAME + " by " + DEVELOPER, page_icon="🏆", layout="wide")
st.title("🏆 " + APP_NAME)
st.caption("NBA and football predictions. Built by " + DEVELOPER + ".")


def run_script(script):
    child_env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    # Streamlit Community Cloud secrets are not automatically inherited by a
    # child Python process. Pass the odds key without printing or persisting it.
    try:
        odds_api_key = st.secrets.get("ODDS_API_KEY")
        if odds_api_key:
            child_env["ODDS_API_KEY"] = str(odds_api_key)
    except Exception:
        pass
    try:
        result = subprocess.run(
            [sys.executable, os.path.join(BASE, script)],
            capture_output=True,
            text=True,
            encoding="utf-8",
            cwd=BASE,
            env=child_env,
            check=False,
        )
    except OSError as error:
        return f"Could not start {script}: {error}"

    output = result.stdout + result.stderr
    if result.returncode:
        output += f"\n{script} exited with code {result.returncode}. See the output above."
    return output or f"{script} finished successfully with no output."


def basketball_view():
    tab_picks, tab_results, tab_update = st.tabs(["Today's picks", "Results", "Update"])

    with tab_picks:
        if st.button("Make predictions for today's games"):
            with st.spinner("Preparing the models and predicting..."):
                output = run_script("predict_today.py")
                injury_output = run_script("injuries.py")
            st.subheader("Predictions")
            st.code(output, language="text")
            st.subheader("Injury report refresh")
            st.code(injury_output, language="text")
        show_basketball_picks(BASE)

    with tab_results:
        st.write("Compares the saved picks with the real results.")
        if st.button("Check results of saved picks"):
            with st.spinner("Checking results..."):
                output = run_script("check_result.py")
            st.code(output, language="text")
        show_results(BASE)

    with tab_update:
        st.write("Downloads the latest finished games (takes about a minute).")
        if st.button("Update game data"):
            with st.spinner("Downloading..."):
                output = run_script("get_all_seasons.py")
            st.code(output, language="text")

        st.divider()
        st.write("Refreshes the injury list.")
        if st.button("Update injury list"):
            with st.spinner("Downloading the injury list..."):
                output = run_script("injuries.py")
            st.code(output, language="text")

        st.divider()
        st.caption(
            "The prediction step checks whether nba_games.csv changed. If it did, the basketball "
            "models retrain automatically before making new picks; if not, the saved models are reused."
        )
        if st.button("Update history, retrain if needed, and refresh picks"):
            for script in ["check_result.py", "get_all_seasons.py",
                           "predict_today.py", "injuries.py"]:
                with st.spinner("Running " + script + " ..."):
                    output = run_script(script)
                st.subheader(script)
                st.code(output, language="text")


def football_view():
    tab_picks, tab_results, tab_update = st.tabs(["Football picks", "Results / performance", "Update"])

    with tab_picks:
        tab_pool, tab_global = st.tabs(["Pool draws", "Global picks"])
        with tab_pool:
            st.caption("Fetch the newest upcoming 49-match coupon from AbleFAST, or use your own list below.")
            if st.button("Fetch next coupon and refresh draw picks"):
                with st.spinner("Fetching the next coupon and rating all fixtures..."):
                    output = run_script("refresh_pool.py")
                st.code(output, language="text")
            st.caption("Manual fallback: pool fixtures are read from pool_list.txt.")
            if st.button("Refresh top two pool draws"):
                with st.spinner("Resolving and rating the pool fixtures..."):
                    pool_file_output = run_script("make_pool_file.py")
                    draw_output = run_script("pool_draws.py")
                st.code(pool_file_output + "\n" + draw_output, language="text")
            show_pool_draws(BASE)

        with tab_global:
            if st.button("Make football predictions"):
                with st.spinner("Preparing the football models and predicting..."):
                    output = run_script("predict_football_today.py")
                st.code(output, language="text")
            show_football_predictions(BASE)

    with tab_results:
        show_football_feedback(BASE)

    with tab_update:
        st.write("Downloads the latest finished football matches (takes a few minutes).")
        if st.button("Update football data"):
            with st.spinner("Downloading the configured football leagues..."):
                output = run_script("get_football_data.py")
            st.code(output, language="text")

        st.caption("Updates actual results, rebuilds the model if the history changed, and saves a fresh forecast.")
        if st.button("Update, retrain, and predict"):
            with st.spinner("Refreshing completed matches..."):
                data_output = run_script("get_football_data.py")
            st.subheader("Football data refresh")
            st.code(data_output, language="text")
            with st.spinner("Training from completed results and making predictions..."):
                prediction_output = run_script("predict_football_today.py")
            st.subheader("Updated football predictions")
            st.code(prediction_output, language="text")

        st.divider()
        st.write("Shows which upcoming matches the free fixtures file currently has.")
        if st.button("Check the fixtures file"):
            with st.spinner("Checking..."):
                output = run_script("check_fixtures.py")
            st.code(output, language="text")


sport = st.radio("Sport", ["Basketball (NBA)", "Football"], horizontal=True)
if sport == "Basketball (NBA)":
    basketball_view()
else:
    football_view()

st.divider()
st.subheader("Phone app")
st.write(
    "Uploads the phone app source and latest prediction and result logs to the "
    "GitHub repository configured in publish.py. Your hosting service controls "
    "when those changes appear online."
)
if st.button("Publish to phone link"):
    with st.spinner("Publishing..."):
        output = run_script("publish.py")
    st.code(output, language="text")

st.divider()
st.caption("© 2026 " + DEVELOPER + ". All rights reserved.")
