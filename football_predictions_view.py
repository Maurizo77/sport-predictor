import os

import pandas as pd
import streamlit as st


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

REQUIRED_COLUMNS = {
    "DATE", "LEAGUE", "HOME", "AWAY", "P_HOME", "P_DRAW", "P_AWAY",
    "P_OVER25", "P_BTTS", "EXP_HOME_GOALS", "EXP_AWAY_GOALS",
}


def show_football_predictions(base):
    path = os.path.join(base, "football_predictions_log.csv")
    if not os.path.exists(path):
        st.info("No football predictions saved yet. Generate predictions when fixtures are available.")
        return

    try:
        view = pd.read_csv(path)
    except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError) as error:
        st.error(f"Could not read the football prediction log: {error}")
        return

    missing = sorted(REQUIRED_COLUMNS - set(view.columns))
    if missing:
        st.error("The football prediction log is missing columns: " + ", ".join(missing))
        return
    if view.empty:
        st.info("No football predictions saved yet.")
        return

    view["DATE"] = pd.to_datetime(view["DATE"], errors="coerce")
    view = view.dropna(subset=["DATE"]).copy()
    if view.empty:
        st.info("There are no dated football predictions to display.")
        return

    for column in ("P_HOME", "P_DRAW", "P_AWAY", "P_OVER25", "P_BTTS",
                   "EXP_HOME_GOALS", "EXP_AWAY_GOALS"):
        view[column] = pd.to_numeric(view[column], errors="coerce")
    view = view.dropna(subset=["P_HOME", "P_DRAW", "P_AWAY"])
    if view.empty:
        st.info("The saved football predictions do not contain usable probabilities.")
        return

    view["League"] = view["LEAGUE"].map(LEAGUE_NAMES).fillna(view["LEAGUE"])
    view["Match"] = view["HOME"].astype(str) + " vs " + view["AWAY"].astype(str)
    view["Most likely"] = view[["P_HOME", "P_DRAW", "P_AWAY"]].idxmax(axis=1).map(
        {"P_HOME": "Home win", "P_DRAW": "Draw", "P_AWAY": "Away win"}
    )
    view["1X2 %"] = view.apply(
        lambda row: " / ".join(
            str(int(round(row[column] * 100))) for column in ("P_HOME", "P_DRAW", "P_AWAY")
        ), axis=1,
    )
    view["Expected goals"] = view.apply(
        lambda row: "–" if pd.isna(row["EXP_HOME_GOALS"]) or pd.isna(row["EXP_AWAY_GOALS"])
        else f"{row['EXP_HOME_GOALS']:.1f} – {row['EXP_AWAY_GOALS']:.1f}", axis=1,
    )
    view["Over 2.5 %"] = view["P_OVER25"].map(
        lambda value: "—" if pd.isna(value) else str(int(round(value * 100))) + "%"
    )
    view["BTTS %"] = view["P_BTTS"].map(
        lambda value: "—" if pd.isna(value) else str(int(round(value * 100))) + "%"
    )

    today = pd.Timestamp.now().normalize()
    upcoming = view[view["DATE"] >= today]
    latest_date = view["DATE"].max().strftime("%Y-%m-%d")
    c1, c2, c3 = st.columns(3)
    c1.metric("Upcoming fixtures", len(upcoming))
    c2.metric("Leagues covered", view["League"].nunique())
    c3.metric("Latest prediction date", latest_date)

    period = st.radio(
        "Fixtures", ["Upcoming", "All saved"], horizontal=True,
        key="football_fixture_period",
    )
    if period == "Upcoming":
        view = upcoming.copy()
        if view.empty:
            st.info("No upcoming fixtures are saved. Choose ‘All saved’ to review older predictions.")
            return

    league_choice = st.selectbox(
        "League", ["All leagues"] + sorted(view["League"].dropna().unique().tolist()),
        key="football_league_filter",
    )
    if league_choice != "All leagues":
        view = view[view["League"] == league_choice]
    view = view.sort_values("DATE", ascending=False).head(60).copy()
    view["DATE"] = view["DATE"].dt.strftime("%Y-%m-%d")

    st.caption("1X2 shows home / draw / away probabilities. Model estimates, not guarantees.")
    st.dataframe(
        view[["DATE", "League", "Match", "Most likely", "1X2 %", "Expected goals",
              "Over 2.5 %", "BTTS %"]],
        hide_index=True,
        use_container_width=True,
    )
