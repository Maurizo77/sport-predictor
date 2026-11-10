import os

import pandas as pd
import streamlit as st

from team_names import full

DEVELOPER = "MAURIZO"
APP_NAME = "Sports Predictor"

BASE = os.path.dirname(os.path.abspath(__file__))

st.set_page_config(page_title=APP_NAME + " by " + DEVELOPER, page_icon="🏀")
st.title("🏀 " + APP_NAME)
st.caption("NBA picks. Built by " + DEVELOPER + ".")

tab_picks, tab_results = st.tabs(["Today's picks", "Results"])

with tab_picks:
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
        st.info("No predictions saved yet.")

with tab_results:
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
        st.info("No checked results yet.")

st.divider()
st.caption("© 2026 " + DEVELOPER + ". All rights reserved.")
st.caption("Data: NBA stats via the nba_api package. Predictions are statistical estimates, not guarantees.")
