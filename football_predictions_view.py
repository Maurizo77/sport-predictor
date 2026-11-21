import os

import pandas as pd
import streamlit as st

from football_feedback import FEEDBACK_FILE, build_feedback, update_feedback_file


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
    if "EVENT_TIME_WAT" not in view:
        view["EVENT_TIME_WAT"] = "Time unavailable"
    view["Kickoff (WAT)"] = view["EVENT_TIME_WAT"].fillna("Time unavailable")
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

    if {"BEST_PICK", "BEST_PICK_PROB"}.issubset(view.columns):
        shortlist = upcoming.dropna(subset=["BEST_PICK", "BEST_PICK_PROB"]).copy()
        if not shortlist.empty:
            shortlist["BEST_PICK_PROB"] = pd.to_numeric(
                shortlist["BEST_PICK_PROB"], errors="coerce"
            )
            shortlist = shortlist.dropna(subset=["BEST_PICK_PROB"])
            shortlist = shortlist.sort_values(
                "BEST_PICK_PROB", ascending=False
            ).head(12)
            if not shortlist.empty:
                st.subheader("Top picks (up to 12)")
                st.caption(
                    "One highest-probability option per match, ranked by the model. "
                    "These are estimates, not sure bets."
                )
                shortlist["Confidence"] = shortlist["BEST_PICK_PROB"].map(
                    lambda value: str(int(round(value * 100))) + "%"
                )
                st.dataframe(
                    shortlist.assign(
                        Date=shortlist["DATE"].dt.strftime("%Y-%m-%d"),
                        Match=shortlist["HOME"].astype(str) + " vs " + shortlist["AWAY"].astype(str),
                        Pick=shortlist["BEST_PICK"],
                    )[["Date", "EVENT_TIME_WAT", "LEAGUE", "Match", "Pick", "Confidence"]],
                    hide_index=True,
                    use_container_width=True,
                )

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

    st.caption(
        "1X2 shows home / draw / away probabilities. Where complete bookmaker prices "
        "were available, the saved 1X2 outcome comes from their no-margin probabilities; "
        "otherwise it comes from the model. Estimates, not guarantees."
    )
    display_columns = [
        "DATE", "Kickoff (WAT)", "League", "Match", "Most likely", "1X2 %",
        "Expected goals", "Over 2.5 %", "BTTS %",
    ]
    if "ODDS_SOURCE" in view.columns:
        view["1X2 source"] = view["ODDS_SOURCE"].fillna("Model")
        display_columns.append("1X2 source")
    st.dataframe(
        view[display_columns],
        hide_index=True,
        use_container_width=True,
    )


def show_pool_draws(base):
    path = os.path.join(base, "pool_picks.csv")
    if not os.path.exists(path):
        st.info("No pool draw picks saved yet. Run the pool draw predictor first.")
        return

    try:
        pool = pd.read_csv(path)
    except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError) as error:
        st.error(f"Could not read the pool draw picks: {error}")
        return

    required = {"DRAW_PICK", "HomeTeam", "AwayTeam", "MODEL", "SCORE", "LEAGUE"}
    missing = sorted(required - set(pool.columns))
    if missing:
        st.error("The pool picks file is missing columns: " + ", ".join(missing))
        return

    draw_mask = pool["DRAW_PICK"].astype(str).str.lower().eq("true")
    picks = pool[draw_mask].copy()
    if picks.empty:
        st.info("No pool draw picks found. Regenerate the pool predictions.")
        return

    picks["MODEL"] = pd.to_numeric(picks["MODEL"], errors="coerce")
    picks["SCORE"] = pd.to_numeric(picks["SCORE"], errors="coerce")
    picks = picks.dropna(subset=["MODEL", "SCORE"]).sort_values("SCORE", ascending=False)
    picks["League"] = picks["LEAGUE"].map(LEAGUE_NAMES).fillna(picks["LEAGUE"])
    picks["Match"] = picks["HomeTeam"].astype(str) + " vs " + picks["AwayTeam"].astype(str)
    picks["Model draw chance"] = (picks["MODEL"] * 100).round(0).astype(int).astype(str) + "%"

    st.subheader("Top pool draw picks")
    st.caption(
        "Ranked using the historical match model and bookmaker draw prices when available. "
        "These are estimates, not guaranteed outcomes."
    )
    columns = ["League", "Match", "Model draw chance"]
    if "BOOK" in picks.columns:
        book = pd.to_numeric(picks["BOOK"], errors="coerce")
        picks["Bookmaker draw chance"] = book.map(
            lambda value: "—" if pd.isna(value) else str(int(round(value * 100))) + "%"
        )
        columns.append("Bookmaker draw chance")
    st.dataframe(
        picks[columns].head(2),
        hide_index=True,
        use_container_width=True,
    )


def show_football_feedback(base):
    feedback = build_feedback(base)
    if not feedback.empty:
        update_feedback_file(base)
    else:
        path = os.path.join(base, FEEDBACK_FILE)
        if not os.path.exists(path):
            st.info(
                "No completed saved forecasts to review yet. Refresh football data "
                "after matches finish to compare forecasts with actual results."
            )
            return
        try:
            feedback = pd.read_csv(path, parse_dates=["DATE"])
        except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError) as error:
            st.error(f"Could not read the football feedback log: {error}")
            return

    if feedback.empty:
        st.info(
            "No completed saved forecasts to review yet. Refresh football data "
            "after matches finish to compare forecasts with actual results."
        )
        return

    st.subheader("How past football predictions performed")
    st.caption(
        "Scored only after actual results are present in the football history. "
        "After the history changes, the next prediction run retrains on those real results; "
        "the model never treats its own guesses as training data."
    )
    c1, c2, c3 = st.columns(3)
    c1.metric("Completed forecasts", len(feedback))
    c2.metric("1X2 accuracy", str(round(feedback["CORRECT"].mean() * 100)) + "%")
    c3.metric("Mean log loss", round(feedback["LOG_LOSS"].mean(), 3))
    st.caption("Accuracy by league is only informative after a meaningful number of completed matches.")

    league_names = feedback["LEAGUE"].map(LEAGUE_NAMES).fillna(feedback["LEAGUE"])
    by_league = feedback.assign(League=league_names).groupby("League").agg(
        Games=("CORRECT", "size"), Correct=("CORRECT", "sum"),
        Accuracy=("CORRECT", "mean"), MeanLogLoss=("LOG_LOSS", "mean"),
    ).reset_index()
    by_league["Accuracy"] = (by_league["Accuracy"] * 100).round(0).astype(int).astype(str) + "%"
    by_league["Mean log loss"] = by_league["MeanLogLoss"].round(3)
    by_league = by_league.drop(columns="MeanLogLoss").sort_values("Games", ascending=False)
    st.dataframe(by_league, hide_index=True, use_container_width=True)

    recent = feedback.head(12).copy()
    recent["Date"] = pd.to_datetime(recent["DATE"], errors="coerce").dt.strftime("%Y-%m-%d")
    recent["Match"] = recent["HOME"].astype(str) + " vs " + recent["AWAY"].astype(str)
    recent["Actual score"] = recent["FTHG"].astype(int).astype(str) + "–" + recent["FTAG"].astype(int).astype(str)
    recent["Outcome check"] = recent.apply(
        lambda row: ("✓ " if row["CORRECT"] else "✗ ") + row["PREDICTED_OUTCOME"] + " / " + row["ACTUAL_OUTCOME"],
        axis=1,
    )
    recent_columns = ["Date", "LEAGUE", "Match", "Outcome check", "Actual score"]
    if "ODDS_SOURCE" in recent.columns:
        recent_columns.append("ODDS_SOURCE")
    st.caption("Most recent completed forecasts")
    st.dataframe(recent[recent_columns], hide_index=True, use_container_width=True)
