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

    st.subheader("Global football predictions")
    st.caption("These fixtures come from the wider supported football feed, independently of the 49-match pool.")

    for column in ("P_HOME", "P_DRAW", "P_AWAY", "P_OVER25", "P_BTTS",
                   "EXP_HOME_GOALS", "EXP_AWAY_GOALS"):
        view[column] = pd.to_numeric(view[column], errors="coerce")
    view = view.dropna(subset=["P_HOME", "P_DRAW", "P_AWAY"])
    if view.empty:
        st.info("The saved football predictions do not contain usable probabilities.")
        return

    model_probability_columns = []
    for source_column, fallback_column, display_column in (
        ("P_HOME_MODEL", "P_HOME", "_MODEL_HOME"),
        ("P_DRAW_MODEL", "P_DRAW", "_MODEL_DRAW"),
        ("P_AWAY_MODEL", "P_AWAY", "_MODEL_AWAY"),
    ):
        if source_column in view.columns:
            view[source_column] = pd.to_numeric(view[source_column], errors="coerce")
            view[display_column] = view[source_column].fillna(view[fallback_column])
        else:
            view[display_column] = view[fallback_column]
        model_probability_columns.append(display_column)

    view["League"] = view["LEAGUE"].map(LEAGUE_NAMES).fillna(view["LEAGUE"])
    view["Match"] = view["HOME"].astype(str) + " vs " + view["AWAY"].astype(str)
    if "EVENT_TIME_WAT" not in view:
        view["EVENT_TIME_WAT"] = "Time unavailable"
    view["Kickoff (WAT)"] = view["EVENT_TIME_WAT"].fillna("Time unavailable")
    view["Most likely"] = view[model_probability_columns].idxmax(axis=1).map(
        {"_MODEL_HOME": "Home win", "_MODEL_DRAW": "Draw", "_MODEL_AWAY": "Away win"}
    )
    view["Model 1X2 %"] = view.apply(
        lambda row: " / ".join(
            str(int(round(row[column] * 100))) for column in model_probability_columns
        ), axis=1,
    )
    probability_sets = {
        "1X2 market reference %": ("P_HOME_MARKET", "P_DRAW_MARKET", "P_AWAY_MARKET"),
    }
    for label, columns in probability_sets.items():
        if all(column in view.columns for column in columns):
            for column in columns:
                view[column] = pd.to_numeric(view[column], errors="coerce")
            view[label] = view.apply(
                lambda row: "—" if row[list(columns)].isna().any() else " / ".join(
                    str(int(round(row[column] * 100))) for column in columns
                ), axis=1,
            )
    for column in ("P_OVER25_MARKET", "P_BTTS_MARKET"):
        if column in view.columns:
            view[column] = pd.to_numeric(view[column], errors="coerce")
    view["Expected goals"] = view.apply(
        lambda row: "–" if pd.isna(row["EXP_HOME_GOALS"]) or pd.isna(row["EXP_AWAY_GOALS"])
        else f"{row['EXP_HOME_GOALS']:.1f} – {row['EXP_AWAY_GOALS']:.1f}", axis=1,
    )
    view["Over 2.5 model %"] = view["P_OVER25"].map(
        lambda value: "—" if pd.isna(value) else str(int(round(value * 100))) + "%"
    )
    view["BTTS model %"] = view["P_BTTS"].map(
        lambda value: "—" if pd.isna(value) else str(int(round(value * 100))) + "%"
    )
    if "P_OVER25_MARKET" in view.columns:
        view["Over 2.5 market %"] = view["P_OVER25_MARKET"].map(
            lambda value: "—" if pd.isna(value) else str(int(round(value * 100))) + "%"
        )
    if "P_BTTS_MARKET" in view.columns:
        view["BTTS market %"] = view["P_BTTS_MARKET"].map(
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
        shortlist_source = upcoming
        if "BEST_PICK_SOURCE" in shortlist_source.columns:
            shortlist_source = shortlist_source[
                shortlist_source["BEST_PICK_SOURCE"].eq("Poisson model")
            ]
        shortlist = shortlist_source.dropna(subset=["BEST_PICK", "BEST_PICK_PROB"]).copy()
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
                    "One highest-probability option per match, ranked using the model's "
                    "own estimates. Bookmaker prices are reference only; estimates are not guarantees."
                )
                shortlist["Estimated chance"] = shortlist["BEST_PICK_PROB"].map(
                    lambda value: str(int(round(value * 100))) + "%"
                )
                shortlist_display = shortlist.assign(
                        Date=shortlist["DATE"].dt.strftime("%Y-%m-%d"),
                        Match=shortlist["HOME"].astype(str) + " vs " + shortlist["AWAY"].astype(str),
                        Pick=shortlist["BEST_PICK"],
                    )
                shortlist_columns = ["Date", "EVENT_TIME_WAT", "LEAGUE", "Match", "Pick", "Estimated chance"]
                if "BEST_PICK_SOURCE" in shortlist_display:
                    shortlist_display["Pick source"] = shortlist_display["BEST_PICK_SOURCE"]
                    shortlist_columns.append("Pick source")
                st.dataframe(
                    shortlist_display[shortlist_columns],
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
        "All percentages are uncertain estimates, never guarantees. Model estimates drive "
        "the predictions and shortlist. Bookmaker 1X2, Over 2.5 and BTTS prices are separate "
        "reference estimates and do not change the model's picks."
    )
    display_columns = [
        "DATE", "Kickoff (WAT)", "League", "Match", "Most likely", "Model 1X2 %",
        "Expected goals", "Over 2.5 model %", "BTTS model %",
    ]
    if "1X2 market reference %" in view.columns:
        display_columns.insert(6, "1X2 market reference %")
    if "P_OVER25_MARKET" in view.columns:
        display_columns.append("Over 2.5 market %")
    if "P_BTTS_MARKET" in view.columns:
        display_columns.append("BTTS market %")
    for source_column, label in (("OVER25_SOURCE", "Over 2.5 odds source"), ("BTTS_SOURCE", "BTTS odds source")):
        if source_column in view.columns and view[source_column].fillna("").astype(str).ne("").any():
            view[label] = view[source_column].replace("", "—").fillna("—")
            display_columns.append(label)
    if "ODDS_SOURCE" in view.columns:
        view["Prediction source"] = view["ODDS_SOURCE"].fillna("Poisson model")
        display_columns.append("Prediction source")
    if "MARKET_1X2_SOURCE" in view.columns and view["MARKET_1X2_SOURCE"].fillna("").astype(str).ne("").any():
        view["1X2 odds reference"] = view["MARKET_1X2_SOURCE"].replace("", "—").fillna("—")
        display_columns.append("1X2 odds reference")
    st.dataframe(
        view[display_columns],
        hide_index=True,
        use_container_width=True,
    )

    auxiliary_pairs = (
        ("EXP_HOME_CORNERS", "EXP_AWAY_CORNERS", "Expected corners (home / away)"),
        ("EXP_HOME_SHOTS", "EXP_AWAY_SHOTS", "Expected shots (home / away)"),
        ("EXP_HOME_SHOTS_ON_TARGET", "EXP_AWAY_SHOTS_ON_TARGET", "Expected shots on target (home / away)"),
        ("EXP_HOME_FOULS", "EXP_AWAY_FOULS", "Expected fouls (home / away)"),
        ("EXP_HOME_OFFSIDES", "EXP_AWAY_OFFSIDES", "Expected offsides (home / away)"),
        ("EXP_HOME_HT_GOALS", "EXP_AWAY_HT_GOALS", "Expected first-half goals (home / away)"),
        ("EXP_HOME_YELLOW_CARDS", "EXP_AWAY_YELLOW_CARDS", "Expected yellow cards (home / away)"),
        ("EXP_HOME_RED_CARDS", "EXP_AWAY_RED_CARDS", "Expected red cards (home / away)"),
    )
    auxiliary_columns = [column for pair in auxiliary_pairs for column in pair[:2]]
    auxiliary_columns += [
        "P_FIRST_HALF_GOAL", "P_FIRST_HALF_2PLUS", "P_ANY_RED_CARD",
        "P_HT_HOME", "P_HT_DRAW", "P_HT_AWAY", "P_OVER85_CORNERS",
        "P_OVER215_SHOTS", "P_OVER75_SHOTS_ON_TARGET", "P_OVER35_YELLOW_CARDS",
        "P_OVER215_FOULS", "P_OVER25_OFFSIDES",
    ]
    if any(column in view.columns for column in auxiliary_columns):
        with st.expander("Additional match-stat predictions (model estimates)"):
            st.caption(
                "These estimates come from historical match statistics; bookmaker odds do not drive them. "
                "Count-based chances use a Poisson approximation and are estimates. "
                "The current history has half-time totals but not goal timestamps, so goals in the first "
                "5, 10, 15, or 20 minutes cannot yet be estimated reliably."
            )
            auxiliary = view.copy()
            for column in auxiliary_columns:
                if column in auxiliary.columns:
                    auxiliary[column] = pd.to_numeric(auxiliary[column], errors="coerce")

            def pair_estimate(row, home_column, away_column):
                if home_column not in auxiliary.columns or away_column not in auxiliary.columns:
                    return "—"
                home_value, away_value = row[home_column], row[away_column]
                if pd.isna(home_value) or pd.isna(away_value):
                    return "—"
                return f"{home_value:.1f} / {away_value:.1f}"

            feature_columns = ["DATE", "League", "Match"]
            for home_column, away_column, label in auxiliary_pairs:
                if home_column in auxiliary.columns and away_column in auxiliary.columns:
                    auxiliary[label] = auxiliary.apply(
                        lambda row: pair_estimate(row, home_column, away_column), axis=1
                    )
                    feature_columns.append(label)
            for column, label in (
                ("P_FIRST_HALF_GOAL", "Chance of 1+ first-half goals"),
                ("P_FIRST_HALF_2PLUS", "Chance of 2+ first-half goals"),
                ("P_ANY_RED_CARD", "Chance of any red card"),
                ("P_OVER85_CORNERS", "Chance of over 8.5 corners"),
                ("P_OVER215_SHOTS", "Chance of over 21.5 shots"),
                ("P_OVER75_SHOTS_ON_TARGET", "Chance of over 7.5 shots on target"),
                ("P_OVER35_YELLOW_CARDS", "Chance of over 3.5 yellow cards"),
                ("P_OVER215_FOULS", "Chance of over 21.5 fouls"),
                ("P_OVER25_OFFSIDES", "Chance of over 2.5 offsides"),
            ):
                if column in auxiliary.columns:
                    auxiliary[label] = auxiliary[column].map(
                        lambda value: "—" if pd.isna(value) else f"{int(round(value * 100))}%"
                    )
                    feature_columns.append(label)
            half_result_columns = ("P_HT_HOME", "P_HT_DRAW", "P_HT_AWAY")
            if all(column in auxiliary.columns for column in half_result_columns):
                auxiliary["Half-time result chance (home / draw / away)"] = auxiliary.apply(
                    lambda row: "—" if row[list(half_result_columns)].isna().any() else " / ".join(
                        f"{int(round(row[column] * 100))}%" for column in half_result_columns
                    ), axis=1,
                )
                feature_columns.append("Half-time result chance (home / draw / away)")
            if "P_ANY_RED_CARD" in auxiliary.columns:
                auxiliary["Chance of no red card"] = auxiliary["P_ANY_RED_CARD"].map(
                    lambda value: "—" if pd.isna(value) else f"{int(round((1 - value) * 100))}%"
                )
                feature_columns.append("Chance of no red card")
            st.dataframe(auxiliary[feature_columns], hide_index=True, use_container_width=True)


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

    required = {"HomeTeam", "AwayTeam", "MODEL", "LEAGUE"}
    missing = sorted(required - set(pool.columns))
    if missing:
        st.error("The pool picks file is missing columns: " + ", ".join(missing))
        return

    picks = pool.copy()
    picks["MODEL"] = pd.to_numeric(picks["MODEL"], errors="coerce")
    picks = picks.dropna(subset=["MODEL"]).sort_values("MODEL", ascending=False).head(2)
    if picks.empty:
        st.info("No pool draw probabilities found. Refresh the current 49-fixture coupon.")
        return
    picks["League"] = picks["LEAGUE"].map(LEAGUE_NAMES).fillna(picks["LEAGUE"])
    picks["Match"] = picks["HomeTeam"].astype(str) + " vs " + picks["AwayTeam"].astype(str)
    picks["Model draw chance"] = (picks["MODEL"] * 100).round(0).astype(int).astype(str) + "%"

    st.subheader("Two draws from the current pool")
    st.caption(
        "These two picks are selected only from the current 49-match pool, ranked by the "
        "model's draw estimates. Bookmaker draw prices are shown for reference only. "
        "Probabilities are estimates, not guarantees."
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
