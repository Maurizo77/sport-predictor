import os

import pandas as pd
import streamlit as st

from results_view import show_results
from team_names import full

DEVELOPER = "ENGR Maurizo"
APP_NAME = "Sports Predictor"

LEAGUE_NAMES = {
    "E0": "England Premier League",
    "SC0": "Scotland Premiership",
    "D1": "Germany Bundesliga",
    "I1": "Italy Serie A",
    "SP1": "Spain La Liga",
    "F1": "France Ligue 1",
    "N1": "Netherlands Eredivisie",
    "B1": "Belgium Pro League",
    "P1": "Portugal Primeira Liga",
    "T1": "Turkey Super Lig",
    "G1": "Greece Super League",
}

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


def best_result(row):
    options = {"Home win": row["P_HOME"], "Draw": row["P_DRAW"], "Away win": row["P_AWAY"]}
    return max(options, key=options.get)


def football_view():
    flog_path = os.path.join(BASE, "football_predictions_log.csv")
    if os.path.exists(flog_path):
        view = pd.read_csv(flog_path)
        view["Match"] = view["HOME"] + " vs " + view["AWAY"]
        view["League"] = view["LEAGUE"].map(LEAGUE_NAMES).fillna(view["LEAGUE"])
        view["Most likely"] = view.apply(best_result, axis=1)
        view["Home %"] = (view["P_HOME"] * 100).round(0)
        view["Draw %"] = (view["P_DRAW"] * 100).round(0)
        view["Away %"] = (view["P_AWAY"] * 100).round(0)
        view["Over 2.5 %"] = (view["P_OVER25"] * 100).round(0)
        view["Both score %"] = (view["P_BTTS"] * 100).round(0)
        view["Expected goals"] = (
            view["EXP_HOME_GOALS"].round(1).astype(str)
            + " - "
            + view["EXP_AWAY_GOALS"].round(1).astype(str)
        )

        league_choice = st.selectbox(
            "League", ["All leagues"] + sorted(view["League"].unique())
        )
        if league_choice != "All leagues":
            view = view[view["League"] == league_choice]
        view = view.sort_values("DATE", ascending=False).head(60)

        st.caption("Probabilities are model estimates, not guarantees.")
        st.dataframe(
            view[["DATE", "League", "Match", "Most likely", "Home %", "Draw %",
                  "Away %", "Over 2.5 %", "Both score %", "Expected goals"]],
            hide_index=True,
        )
    else:
        st.info("No football predictions yet. They will appear here soon.")


sport = st.radio("Sport", ["Basketball (NBA)", "Football"], horizontal=True)
if sport == "Basketball (NBA)":
    basketball_view()
else:
    football_view()

st.divider()
st.caption("© 2026 " + DEVELOPER + ". All rights reserved.")
st.caption("Predictions are statistical estimates, not guarantees.")