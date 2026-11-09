import os
import subprocess
import sys

import pandas as pd
import streamlit as st

from team_names import full

BASE = os.path.dirname(os.path.abspath(__file__))

st.set_page_config(page_title="Sports Predictor", page_icon="🏀")
st.title("🏀 Sports Predictor")


def run_script(script):
    result = subprocess.run(
        [sys.executable, script],
        capture_output=True,
        text=True,
        encoding="utf-8",
        cwd=BASE,
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
    )
    return result.stdout + result.stderr


tab_picks, tab_results, tab_update = st.tabs(["Today's picks", "Results", "Update"])

with tab_picks:
    if st.button("Make predictions for today's games"):
        with st.spinner("Training the models and predicting..."):
            output = run_script("predict_today.py")
        st.code(output, language="text")

    log_path = os.path.join(BASE, "predictions_log.csv")
    if os.path.exists(log_path):
        log = pd.read_csv(log_path, dtype={"GAME_ID": str})
        latest = log[log["DATE"] == log["DATE"].max()].copy()
        latest["Game"] = latest["AWAY"].map(full) + " @ " + latest["HOME"].map(full)
        latest["Pick"] = latest["PICK"].map(full)
        latest["Confidence %"] = (
            latest["HOME_WIN_PROB"].apply(lambda p: max(p, 1 - p)) * 100
        ).round(0)
        latest["Predicted total"] = latest["PRED_TOTAL"]
        st.caption("Latest saved predictions, for " + str(log["DATE"].max()))
        st.dataframe(
            latest[["Game", "Pick", "Confidence %", "Predicted total"]],
            hide_index=True,
        )
    else:
        st.info("No predictions saved yet. Click the button above.")

with tab_results:
    st.write("Compares the saved picks with the real results.")
    if st.button("Check results of saved picks"):
        with st.spinner("Checking results..."):
            run_script("check_result.py")

    results_path = os.path.join(BASE, "results_log.csv")
    if os.path.exists(results_path):
        res = pd.read_csv(results_path, dtype={"GAME_ID": str})
        res["Result"] = res["CORRECT"].map({True: "✔", False: "✘"})
        res["Total error"] = (res["PRED_TOTAL"] - res["REAL_TOTAL"]).abs().round(1)

        c1, c2, c3 = st.columns(3)
        c1.metric("Games checked", len(res))
        c2.metric("Correct", int(res["CORRECT"].sum()))
        c3.metric("Accuracy", str(round(res["CORRECT"].mean() * 100)) + "%")
        st.metric("Average total error (points)", round(res["Total error"].mean(), 1))

        st.dataframe(
            res[["DATE", "GAME", "PICK", "WINNER", "Result",
                 "PRED_TOTAL", "REAL_TOTAL", "Total error"]],
            hide_index=True,
        )
    else:
        st.info("No checked results yet. Click the button above.")

with tab_update:
    st.write("Downloads the latest finished games (takes about a minute).")
    if st.button("Update game data"):
        with st.spinner("Downloading..."):
            output = run_script("get_all_seasons.py")
        st.code(output, language="text")

    st.divider()
    if st.button("Run everything: check results, update data, predict"):
        for script in ["check_result.py", "get_all_seasons.py", "predict_today.py"]:
            with st.spinner("Running " + script + " ..."):
                output = run_script(script)
            st.subheader(script)
            st.code(output, language="text")