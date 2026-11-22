import os

import pandas as pd
import streamlit as st

from football_predictions_view import (
    show_football_feedback,
    show_football_predictions,
    show_pool_draws,
)
from results_view import show_results
from team_names import full

DEVELOPER = "ENGR Maurizo"
APP_NAME = "Sports Predictor"

BASE = os.path.dirname(os.path.abspath(__file__))

st.set_page_config(page_title=APP_NAME + " by " + DEVELOPER, page_icon="🏆")
st.title("🏆 " + APP_NAME)
st.caption("NBA and football predictions. Built by " + DEVELOPER + ".")


def basketball_view():
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
        show_results(BASE)


def football_view():
    tab_picks, tab_results = st.tabs(["Football picks", "Results / performance"])
    with tab_picks:
        show_pool_draws(BASE)
        st.divider()
        show_football_predictions(BASE)
    with tab_results:
        show_football_feedback(BASE)


sport = st.radio("Sport", ["Basketball (NBA)", "Football"], horizontal=True)
if sport == "Basketball (NBA)":
    basketball_view()
else:
    football_view()

st.divider()
st.caption("© 2026 " + DEVELOPER + ". All rights reserved.")
st.caption("Predictions are statistical estimates, not guarantees.")
