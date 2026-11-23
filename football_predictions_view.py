import os

import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import poisson
from sklearn.isotonic import IsotonicRegression

from football_feedback import FEEDBACK_FILE, build_feedback, update_feedback_file
from football_market_features import FEATURE_GROUPS
from event_time import FOOTBALL_FEED_TIMEZONE, NIGERIA_TIMEZONE


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
CALIBRATION_MIN_SAMPLES = 50


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
    view["Home win or draw (1X) %"] = (
        view["_MODEL_HOME"] + view["_MODEL_DRAW"]
    ).map(lambda value: f"{int(round(value * 100))}%")
    view["Draw or away win (X2) %"] = (
        view["_MODEL_DRAW"] + view["_MODEL_AWAY"]
    ).map(lambda value: f"{int(round(value * 100))}%")
    view["Either team to win (12) %"] = (
        view["_MODEL_HOME"] + view["_MODEL_AWAY"]
    ).map(lambda value: f"{int(round(value * 100))}%")
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
        lambda row: "—" if pd.isna(row["EXP_HOME_GOALS"]) or pd.isna(row["EXP_AWAY_GOALS"])
        else f"Home {row['EXP_HOME_GOALS']:.1f} / Away {row['EXP_AWAY_GOALS']:.1f}", axis=1,
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

    derived_probabilities = {
        "P_HOME_OR_DRAW": view["_MODEL_HOME"] + view["_MODEL_DRAW"],
        "P_DRAW_OR_AWAY": view["_MODEL_DRAW"] + view["_MODEL_AWAY"],
        "P_HOME_OR_AWAY": view["_MODEL_HOME"] + view["_MODEL_AWAY"],
        "P_UNDER25": 1 - view["P_OVER25"],
        "P_BTTS_NO": 1 - view["P_BTTS"],
    }
    total_goal_rate = view["EXP_HOME_GOALS"] + view["EXP_AWAY_GOALS"]
    derived_probabilities["P_OVER15"] = pd.Series(poisson.sf(1, total_goal_rate), index=view.index)
    derived_probabilities["P_UNDER15"] = 1 - derived_probabilities["P_OVER15"]
    derived_probabilities["P_OVER35"] = pd.Series(poisson.sf(3, total_goal_rate), index=view.index)
    derived_probabilities["P_UNDER35"] = 1 - derived_probabilities["P_OVER35"]
    for target, source in {
        "P_NO_FIRST_HALF_GOAL": "P_FIRST_HALF_GOAL",
        "P_FIRST_HALF_UNDER15": "P_FIRST_HALF_2PLUS",
        "P_UNDER85_CORNERS": "P_OVER85_CORNERS",
        "P_UNDER215_SHOTS": "P_OVER215_SHOTS",
        "P_UNDER75_SHOTS_ON_TARGET": "P_OVER75_SHOTS_ON_TARGET",
        "P_UNDER35_YELLOW_CARDS": "P_OVER35_YELLOW_CARDS",
        "P_UNDER215_FOULS": "P_OVER215_FOULS",
        "P_UNDER25_OFFSIDES": "P_OVER25_OFFSIDES",
        "P_NO_RED_CARD": "P_ANY_RED_CARD",
    }.items():
        if source in view.columns:
            derived_probabilities[target] = 1 - pd.to_numeric(view[source], errors="coerce")
    for target, fallback in derived_probabilities.items():
        fallback = pd.to_numeric(fallback, errors="coerce")
        if target in view.columns:
            view[target] = pd.to_numeric(view[target], errors="coerce").fillna(fallback)
        else:
            view[target] = fallback

    now_wat = pd.Timestamp.now(tz=NIGERIA_TIMEZONE)
    today = pd.Timestamp.now(tz=FOOTBALL_FEED_TIMEZONE).normalize().tz_localize(None)
    historical_feedback = build_feedback(base)
    completed_keys = set()
    if not historical_feedback.empty:
        completed_dates = pd.to_datetime(historical_feedback["DATE"], errors="coerce").dt.strftime("%Y-%m-%d")
        completed_keys = set(zip(
            completed_dates,
            historical_feedback["LEAGUE"].astype(str),
            historical_feedback["HOME"].astype(str),
            historical_feedback["AWAY"].astype(str),
        ))
    fixture_keys = list(zip(
        view["DATE"].dt.strftime("%Y-%m-%d"),
        view["LEAGUE"].astype(str), view["HOME"].astype(str), view["AWAY"].astype(str),
    ))
    has_not_started = view["DATE"] > today
    same_day = view["DATE"] == today
    kickoff_text = view["Kickoff (WAT)"].astype(str).str.replace(" WAT", "", regex=False)
    kickoff_naive = pd.to_datetime(kickoff_text, format="%Y-%m-%d %H:%M", errors="coerce")
    kickoff_wat = kickoff_naive.dt.tz_localize(NIGERIA_TIMEZONE)
    has_not_started |= same_day & (kickoff_wat.isna() | (kickoff_wat > now_wat))
    not_completed = pd.Series([key not in completed_keys for key in fixture_keys], index=view.index)
    upcoming = view[has_not_started & not_completed]
    latest_date = view["DATE"].max().strftime("%Y-%m-%d")
    c1, c2, c3 = st.columns(3)
    c1.metric("Fixtures screened", len(upcoming))
    c2.metric("Prediction features", len(FEATURE_GROUPS))
    c3.metric("Latest prediction date", latest_date)

    if not historical_feedback.empty:
        historical_feedback["DATE"] = pd.to_datetime(historical_feedback["DATE"], errors="coerce")
        historical_feedback = historical_feedback[historical_feedback["DATE"] < today]
    feature_calibrators = {}
    calibration_counts = {}
    for _group, markets in FEATURE_GROUPS.items():
        for key, _label, _probability_column, _actual_event in markets:
            probability_column = "FEATURE_PROB_" + key
            hit_column = "FEATURE_HIT_" + key
            if probability_column not in historical_feedback or hit_column not in historical_feedback:
                calibration_counts[key] = 0
                continue
            historical_probability = pd.to_numeric(
                historical_feedback[probability_column], errors="coerce"
            )
            historical_hit = historical_feedback[hit_column].astype("boolean")
            valid = historical_probability.notna() & historical_hit.notna()
            calibration_counts[key] = int(valid.sum())
            if (
                valid.sum() >= CALIBRATION_MIN_SAMPLES
                and historical_probability.loc[valid].nunique() >= 5
                and historical_hit.loc[valid].nunique() == 2
            ):
                calibrator = IsotonicRegression(
                    y_min=0, y_max=1, out_of_bounds="clip"
                )
                calibrator.fit(
                    historical_probability.loc[valid].to_numpy(dtype=float),
                    historical_hit.loc[valid].astype(int).to_numpy(),
                )
                feature_calibrators[key] = calibrator

    adjusted_probabilities = {}
    calibration_applied = {}
    for _group, markets in FEATURE_GROUPS.items():
        for key, _label, probability_column, _actual_event in markets:
            if probability_column not in upcoming.columns:
                continue
            raw = pd.to_numeric(upcoming[probability_column], errors="coerce")
            adjusted = raw.copy()
            calibrator = feature_calibrators.get(key)
            valid = raw.notna()
            if calibrator is not None and valid.any():
                adjusted.loc[valid] = calibrator.predict(raw.loc[valid].to_numpy(dtype=float))
            adjusted_probabilities[key] = adjusted
            calibration_applied[key] = calibrator is not None

    def normalize_feature_group(keys):
        if not all(key in adjusted_probabilities for key in keys):
            return
        values = pd.concat([adjusted_probabilities[key].rename(key) for key in keys], axis=1)
        totals = values.sum(axis=1, min_count=len(keys))
        normalized = values.div(totals.where(totals > 0), axis=0)
        applied = any(calibration_applied.get(key, False) for key in keys)
        for key in keys:
            adjusted_probabilities[key] = normalized[key]
            calibration_applied[key] = applied

    normalize_feature_group(("home_win", "draw", "away_win"))
    normalize_feature_group(("ht_home", "ht_draw", "ht_away"))
    for target, parts in {
        "home_draw": ("home_win", "draw"),
        "draw_away": ("draw", "away_win"),
        "home_away": ("home_win", "away_win"),
    }.items():
        if all(part in adjusted_probabilities for part in parts):
            adjusted_probabilities[target] = sum(adjusted_probabilities[part] for part in parts)
            calibration_applied[target] = any(calibration_applied.get(part, False) for part in parts)
    for positive, negative in (
        ("over15", "under15"), ("over25", "under25"), ("over35", "under35"),
        ("btts_yes", "btts_no"), ("ht_goal", "ht_no_goal"),
        ("ht_2plus", "ht_under15"), ("corners_over85", "corners_under85"),
        ("shots_over215", "shots_under215"), ("sot_over75", "sot_under75"),
        ("yellow_over35", "yellow_under35"), ("red_yes", "red_no"),
        ("fouls_over215", "fouls_under215"), ("offsides_over25", "offsides_under25"),
    ):
        if positive in adjusted_probabilities and negative in adjusted_probabilities:
            adjusted_probabilities[negative] = 1 - adjusted_probabilities[positive]
            calibration_applied[negative] = calibration_applied.get(positive, False)

    confidence_floor = st.slider(
        "Minimum model confidence in feature lists", min_value=50, max_value=95,
        value=70, step=5, key="football_feature_confidence",
    ) / 100
    st.caption(
        "Open a feature heading to see qualifying upcoming selections ranked by confidence. "
        "One match can appear under several features. Confidence is an estimate, not a guarantee."
    )
    overall_candidates = []
    for group, markets in FEATURE_GROUPS.items():
        for key, label, probability_column, _actual_event in markets:
            if probability_column not in upcoming.columns:
                continue
            for row_index, probability in pd.to_numeric(
                upcoming[probability_column], errors="coerce"
            ).dropna().items():
                raw_probability = float(probability)
                adjusted_probability = float(adjusted_probabilities[key].loc[row_index])
                row = upcoming.loc[row_index]
                overall_candidates.append({
                    "DATE": row["DATE"], "LEAGUE": row["LEAGUE"],
                    "HOME": row["HOME"], "AWAY": row["AWAY"],
                    "Kickoff (WAT)": row["Kickoff (WAT)"],
                    "Match": row["Match"], "Feature": group,
                    "Selection": label, "Model estimate": raw_probability,
                    "Adjusted confidence": adjusted_probability,
                    "Completed feature forecasts": calibration_counts.get(key, 0),
                    "Calibration applied": calibration_applied.get(key, False),
                })
    if overall_candidates:
        overall = pd.DataFrame(overall_candidates).sort_values(
            "Adjusted confidence", ascending=False, kind="mergesort"
        )
        overall = overall.drop_duplicates(
            ["DATE", "LEAGUE", "HOME", "AWAY"], keep="first"
        ).head(12).copy()
        overall.insert(0, "Rank", range(1, len(overall) + 1))
        overall["Date"] = pd.to_datetime(overall["DATE"]).dt.strftime("%Y-%m-%d")
        overall["Model estimate"] = overall["Model estimate"].map(lambda value: f"{value:.0%}")
        overall["Adjusted confidence"] = overall["Adjusted confidence"].map(lambda value: f"{value:.0%}")
        overall["History"] = overall.apply(
            lambda row: f"{row['Completed feature forecasts']} scored; "
            + ("calibrated" if row["Calibration applied"] else "raw estimate used"), axis=1
        )
        st.subheader(f"Top {len(overall)} upcoming fixture picks (up to 12)")
        st.caption(
            "One strongest selection per fixture, ranked by adjusted estimate. This summary can include picks "
            f"below the {confidence_floor:.0%} feature-list filter so more available games appear. When a feature has at least "
            f"{CALIBRATION_MIN_SAMPLES} completed forecasts with varied confidence and both outcomes, its probability is adjusted using prior outcomes; "
            "otherwise the model estimate is shown unchanged. Bookmaker odds do not affect this list."
        )
        st.dataframe(
            overall[[
                "Rank", "Date", "Kickoff (WAT)", "LEAGUE", "Match", "Feature", "Selection",
                "Model estimate", "Adjusted confidence", "History",
            ]],
            hide_index=True, use_container_width=True,
        )
    else:
        st.info("No upcoming fixture predictions are available.")

    for group, markets in FEATURE_GROUPS.items():
        selections = []
        for key, label, probability_column, _actual_event in markets:
            if probability_column not in upcoming.columns:
                continue
            probabilities = pd.to_numeric(upcoming[probability_column], errors="coerce")
            adjusted = adjusted_probabilities[key]
            was_calibrated = calibration_applied.get(key, False)
            qualified_mask = (probabilities >= confidence_floor) & (adjusted >= confidence_floor)
            qualified = upcoming.loc[qualified_mask].copy()
            if qualified.empty:
                continue
            qualified["Selection"] = label
            qualified["Model estimate"] = probabilities.loc[qualified.index]
            qualified["Adjusted confidence"] = adjusted.loc[qualified.index]
            qualified["History"] = (
                f"{calibration_counts.get(key, 0)} scored; "
                + ("calibrated" if was_calibrated else "raw estimate used")
            )
            selections.append(qualified[[
                "DATE", "Kickoff (WAT)", "League", "Match", "Selection",
                "Model estimate", "Adjusted confidence", "History",
            ]])
        feature_rows = pd.concat(selections, ignore_index=True) if selections else pd.DataFrame()
        if not feature_rows.empty:
            feature_rows = feature_rows.sort_values("Adjusted confidence", ascending=False)
            feature_rows["Date"] = pd.to_datetime(feature_rows["DATE"]).dt.strftime("%Y-%m-%d")
            feature_rows["Model estimate"] = feature_rows["Model estimate"].map(lambda value: f"{value:.0%}")
            feature_rows["Adjusted confidence"] = feature_rows["Adjusted confidence"].map(lambda value: f"{value:.0%}")
            match_count = feature_rows[["DATE", "League", "Match"]].drop_duplicates().shape[0]
        else:
            match_count = 0
        with st.expander(f"{group} — {len(feature_rows)} selections across {match_count} matches"):
            if feature_rows.empty:
                st.info("No upcoming selection meets the current confidence filter for this feature.")
            else:
                st.dataframe(
                    feature_rows[[
                        "Date", "Kickoff (WAT)", "League", "Match", "Selection",
                        "Model estimate", "Adjusted confidence", "History",
                    ]],
                    hide_index=True, use_container_width=True,
                )

    if not st.checkbox("Show full fixture table and optional match-stat estimates", value=False):
        return
    period = st.radio("Fixtures", ["Upcoming", "All saved"], horizontal=True, key="football_fixture_period")
    view = upcoming.copy() if period == "Upcoming" else view.copy()
    if view.empty:
        st.info("No fixtures are available in this view.")
        return

    league_choice = st.selectbox(
        "League", ["All leagues"] + sorted(view["League"].dropna().unique().tolist()),
        key="football_league_filter",
    )
    if league_choice != "All leagues":
        view = view[view["League"] == league_choice]
    view = view.sort_values("DATE", ascending=False)
    view = view.head(60).copy()
    view["DATE"] = view["DATE"].dt.strftime("%Y-%m-%d")

    st.caption(
        "All percentages are uncertain estimates, never guarantees. Feature lists use model estimates. "
        "Bookmaker 1X2, Over 2.5 and BTTS prices are separate "
        "reference estimates and do not change the model's picks. 1X2 is home/draw/away; "
        "1X means home-or-draw, X2 means draw-or-away, and 12 means either team wins."
    )
    display_columns = [
        "DATE", "Kickoff (WAT)", "League", "Match", "Most likely", "Model 1X2 %",
        "Home win or draw (1X) %", "Draw or away win (X2) %",
        "Either team to win (12) %", "Expected goals (home / away)", "Over 2.5 model %", "BTTS model %",
    ]
    view["Expected goals (home / away)"] = view["Expected goals"]
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

    for column in feedback.columns:
        if column == "BEST_PICK_CORRECT" or column.startswith("FEATURE_HIT_"):
            feedback[column] = feedback[column].map(
                lambda value: pd.NA if pd.isna(value) else str(value).strip().lower() in {"true", "1"}
            ).astype("boolean")

    st.subheader("Football model performance")
    st.caption(
        "Only completed matches with actual scores in the refreshed football history are counted. "
        "✓ means the model's most likely 1X2 outcome was correct; ✗ means it was wrong. "
        "Use the sample counts alongside the percentages: a few matches are not enough to judge accuracy. "
        "Lower log loss, Brier score, and mean absolute error are better."
    )
    completed = len(feedback)
    goal_mae = pd.to_numeric(feedback["TOTAL_GOAL_ERROR"], errors="coerce").mean()
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Completed matches", completed)
    c2.metric("1X2 correct", f"{feedback['CORRECT'].mean():.0%}", f"{int(feedback['CORRECT'].sum())} of {completed}")
    c3.metric("Avg total-goal error", "—" if pd.isna(goal_mae) else f"{goal_mae:.2f} goals")
    c4.metric("Mean log loss", f"{feedback['LOG_LOSS'].mean():.3f}")
    c5.metric("Mean Brier score", f"{feedback['BRIER_SCORE'].mean():.3f}")

    market_columns = ["P_HOME_MARKET", "P_DRAW_MARKET", "P_AWAY_MARKET"]
    if set(market_columns + ["ACTUAL_OUTCOME"]).issubset(feedback.columns):
        benchmark = feedback.copy()
        market_probabilities = benchmark[market_columns].apply(pd.to_numeric, errors="coerce")
        model_columns = ["P_HOME", "P_DRAW", "P_AWAY"]
        model_probabilities = benchmark[model_columns].apply(pd.to_numeric, errors="coerce")
        actual_indices = benchmark["ACTUAL_OUTCOME"].map(
            {"Home win": 0, "Draw": 1, "Away win": 2}
        )
        available = market_probabilities.notna().all(axis=1) & actual_indices.notna()
        if available.any():
            market_values = np.clip(market_probabilities.loc[available].to_numpy(dtype=float), 1e-6, 1)
            market_values /= market_values.sum(axis=1, keepdims=True)
            model_values = np.clip(model_probabilities.loc[available].to_numpy(dtype=float), 1e-6, 1)
            model_values /= model_values.sum(axis=1, keepdims=True)
            actual = actual_indices.loc[available].to_numpy(dtype=int)
            actual_one_hot = np.eye(3)[actual]
            benchmark_rows = []
            for label, values in (
                ("Model", model_values), ("Bookmaker reference", market_values),
            ):
                predicted = values.argmax(axis=1)
                benchmark_rows.append({
                    "Source": label, "Same fixtures": len(values),
                    "Correct": int((predicted == actual).sum()),
                    "Accuracy": float((predicted == actual).mean()),
                    "Log loss": float(-np.log(values[np.arange(len(values)), actual]).mean()),
                    "Brier score": float(((values - actual_one_hot) ** 2).sum(axis=1).mean()),
                })
            benchmark_table = pd.DataFrame(benchmark_rows)
            benchmark_table["Correct / scored"] = benchmark_table.apply(
                lambda row: f"{int(row['Correct'])} / {int(row['Same fixtures'])}", axis=1
            )
            benchmark_table["Accuracy"] = benchmark_table["Accuracy"].map(lambda value: f"{value:.0%}")
            benchmark_table["Log loss"] = benchmark_table["Log loss"].map(lambda value: f"{value:.3f}")
            benchmark_table["Brier score"] = benchmark_table["Brier score"].map(lambda value: f"{value:.3f}")
            st.markdown("**Model vs bookmaker reference (same completed fixtures)**")
            st.caption(
                "This is a results-only benchmark on fixtures with complete odds. "
                "Bookmaker probabilities do not choose or change model predictions. "
                "Lower log loss and Brier score are better; use the sample size when comparing."
            )
            st.dataframe(
                benchmark_table[["Source", "Correct / scored", "Accuracy", "Log loss", "Brier score"]],
                hide_index=True, use_container_width=True,
            )

    score_rows = []
    score_rows.append({
        "Prediction market": "Full-time result (1X2)", "Matches scored": completed,
        "Correct": int(feedback["CORRECT"].sum()), "Accuracy": feedback["CORRECT"].mean(),
    })
    if "BEST_PICK_CORRECT" in feedback:
        best = feedback["BEST_PICK_CORRECT"].dropna().astype(bool)
        if not best.empty:
            score_rows.append({
                "Prediction market": "Saved highest-probability pick (market varies by forecast)",
                "Matches scored": len(best), "Correct": int(best.sum()), "Accuracy": best.mean(),
            })
    confidence_floor = float(st.session_state.get("football_feature_confidence", 70)) / 100
    feature_history = {}
    for group, markets in FEATURE_GROUPS.items():
        group_history = []
        for key, label, _probability_column, _actual_event in markets:
            prob_column = "FEATURE_PROB_" + key
            hit_column = "FEATURE_HIT_" + key
            if prob_column not in feedback or hit_column not in feedback:
                continue
            probabilities = pd.to_numeric(feedback[prob_column], errors="coerce")
            eligible = (probabilities >= confidence_floor) & feedback[hit_column].notna()
            scored = feedback.loc[eligible].copy()
            if scored.empty:
                continue
            hits = scored[hit_column].astype(bool)
            score_rows.append({
                "Prediction market": f"{group}: {label}",
                "Matches scored": len(scored), "Correct": int(hits.sum()),
                "Accuracy": hits.mean(),
            })
            scored["Selection"] = label
            scored["Model confidence"] = probabilities.loc[scored.index]
            scored["Feature result"] = hits.map({True: "✓ Correct", False: "✗ Incorrect"})
            scored["Actual score"] = (
                scored["FTHG"].astype(int).astype(str) + "–" + scored["FTAG"].astype(int).astype(str)
            )
            group_history.append(scored)
        feature_history[group] = pd.concat(group_history, ignore_index=False) if group_history else pd.DataFrame()
    if score_rows:
        performance = pd.DataFrame(score_rows)
        performance["Correct / scored"] = performance.apply(
            lambda row: f"{int(row['Correct'])} / {int(row['Matches scored'])}", axis=1
        )
        performance["Accuracy"] = performance["Accuracy"].map(lambda value: f"{value:.0%}")
        st.markdown("**Accuracy by prediction market**")
        st.dataframe(
            performance[["Prediction market", "Correct / scored", "Accuracy"]],
            hide_index=True, use_container_width=True,
        )

    st.markdown("**Completed selections by feature**")
    st.caption(
        f"Each row is scored against its own result when available. The accuracy tables use the current "
        f"{confidence_floor:.0%} confidence filter, so one match may count under multiple features."
    )
    for group, _markets in FEATURE_GROUPS.items():
        history = feature_history.get(group, pd.DataFrame())
        if history.empty:
            with st.expander(f"{group} — no completed selections at {confidence_floor:.0%}+"):
                st.info("Completed match data for this feature will appear after the results are refreshed.")
            continue
        correct_count = int((history["Feature result"] == "✓ Correct").sum())
        with st.expander(
            f"{group} — {correct_count} of {len(history)} correct at {confidence_floor:.0%}+"
        ):
            scored_history = history.copy()
            scored_history["Date"] = pd.to_datetime(scored_history["DATE"], errors="coerce").dt.strftime("%Y-%m-%d")
            scored_history["Model confidence"] = scored_history["Model confidence"].map(lambda value: f"{value:.0%}")
            scored_history["Match"] = scored_history["HOME"].astype(str) + " vs " + scored_history["AWAY"].astype(str)
            st.dataframe(
                scored_history[["Date", "LEAGUE", "Match", "Selection", "Model confidence", "Feature result", "Actual score"]],
                hide_index=True, use_container_width=True,
            )

    error_labels = {
        "HOME_GOAL_ERROR": "Home goals", "AWAY_GOAL_ERROR": "Away goals",
        "TOTAL_GOAL_ERROR": "Total goals", "ERROR_EXP_HOME_CORNERS": "Home corners",
        "ERROR_EXP_AWAY_CORNERS": "Away corners", "ERROR_EXP_HOME_SHOTS": "Home shots",
        "ERROR_EXP_AWAY_SHOTS": "Away shots", "ERROR_EXP_HOME_SHOTS_ON_TARGET": "Home shots on target",
        "ERROR_EXP_AWAY_SHOTS_ON_TARGET": "Away shots on target", "ERROR_EXP_HOME_FOULS": "Home fouls",
        "ERROR_EXP_AWAY_FOULS": "Away fouls", "ERROR_EXP_HOME_OFFSIDES": "Home offsides",
        "ERROR_EXP_AWAY_OFFSIDES": "Away offsides", "ERROR_EXP_HOME_HT_GOALS": "Home first-half goals",
        "ERROR_EXP_AWAY_HT_GOALS": "Away first-half goals", "ERROR_EXP_HOME_YELLOW_CARDS": "Home yellow cards",
        "ERROR_EXP_AWAY_YELLOW_CARDS": "Away yellow cards", "ERROR_EXP_HOME_RED_CARDS": "Home red cards",
        "ERROR_EXP_AWAY_RED_CARDS": "Away red cards",
    }
    error_rows = []
    for column, label in error_labels.items():
        if column in feedback:
            errors = pd.to_numeric(feedback[column], errors="coerce").dropna()
            if not errors.empty:
                error_rows.append({
                    "Estimate": label, "Matches scored": len(errors),
                    "Mean absolute error": round(errors.mean(), 2),
                })
    if error_rows:
        with st.expander("How close were the model's number estimates?"):
            st.caption("Mean absolute error: the average distance between the model estimate and the actual count.")
            st.dataframe(pd.DataFrame(error_rows), hide_index=True, use_container_width=True)

    if {"BEST_PICK_PROB", "BEST_PICK_CORRECT"}.issubset(feedback.columns):
        confidence = feedback.assign(
            CONFIDENCE=pd.to_numeric(feedback["BEST_PICK_PROB"], errors="coerce"),
            CONFIDENCE_CORRECT=feedback["BEST_PICK_CORRECT"].astype("boolean"),
        ).dropna(subset=["CONFIDENCE", "CONFIDENCE_CORRECT"])
        high_confidence = confidence[confidence["CONFIDENCE"] >= confidence_floor]
        if not high_confidence.empty:
            high_hits = high_confidence["CONFIDENCE_CORRECT"].astype(bool)
            st.metric(
                f"Saved picks estimated at {confidence_floor:.0%}+",
                f"{high_hits.mean():.0%} correct",
                f"{int(high_hits.sum())} of {len(high_hits)} completed picks",
            )
    else:
        confidence = feedback.assign(
            CONFIDENCE=feedback[["P_HOME", "P_DRAW", "P_AWAY"]].max(axis=1),
            CONFIDENCE_CORRECT=feedback["CORRECT"],
        )
    confidence_bins = [0.0, 0.5, 0.6, 0.7, 0.8, 1.01]
    confidence_labels = ["Below 50%", "50–59%", "60–69%", "70–79%", "80%+"]
    confidence["Confidence band"] = pd.cut(
        confidence["CONFIDENCE"], bins=confidence_bins, labels=confidence_labels,
        right=False, include_lowest=True,
    )
    by_confidence = confidence.groupby("Confidence band", observed=True)["CONFIDENCE_CORRECT"].agg(
        Matches="size", Correct="sum", Accuracy="mean"
    ).reset_index()
    if not by_confidence.empty:
        by_confidence["Correct / scored"] = by_confidence.apply(
            lambda row: f"{int(row['Correct'])} / {int(row['Matches'])}", axis=1
        )
        by_confidence["Accuracy"] = by_confidence["Accuracy"].map(lambda value: f"{value:.0%}")
        with st.expander("Accuracy by model confidence"):
            st.dataframe(by_confidence[["Confidence band", "Correct / scored", "Accuracy"]], hide_index=True)

    league_names = feedback["LEAGUE"].map(LEAGUE_NAMES).fillna(feedback["LEAGUE"])
    by_league = feedback.assign(League=league_names).groupby("League").agg(
        Matches=("CORRECT", "size"), Correct=("CORRECT", "sum"),
        Accuracy=("CORRECT", "mean"), MeanLogLoss=("LOG_LOSS", "mean"),
    ).reset_index()
    by_league["Correct / scored"] = by_league.apply(
        lambda row: f"{int(row['Correct'])} / {int(row['Matches'])}", axis=1
    )
    by_league["Accuracy"] = by_league["Accuracy"].map(lambda value: f"{value:.0%}")
    by_league["Mean log loss"] = by_league["MeanLogLoss"].round(3)
    by_league = by_league.drop(columns=["Correct", "MeanLogLoss"]).sort_values("Matches", ascending=False)
    with st.expander("1X2 performance by league"):
        st.dataframe(by_league, hide_index=True, use_container_width=True)

    # Surface repeatable weak spots from the model's own results. Require a
    # minimum sample to avoid presenting a handful of misses as a real pattern.
    pattern_history = feedback.copy()
    pattern_history["Confidence"] = pattern_history[["P_HOME", "P_DRAW", "P_AWAY"]].max(axis=1)
    pattern_history["Confidence band"] = pd.cut(
        pattern_history["Confidence"], bins=[0, .6, .7, .8, 1.01],
        labels=["Below 60%", "60–69%", "70–79%", "80%+"], right=False,
    )
    patterns = pattern_history.groupby(
        ["LEAGUE", "PREDICTED_OUTCOME", "Confidence band"], observed=True
    ).agg(
        Matches=("CORRECT", "size"), Correct=("CORRECT", "sum"),
        Accuracy=("CORRECT", "mean"), MeanLogLoss=("LOG_LOSS", "mean"),
    ).reset_index()
    patterns = patterns[patterns["Matches"] >= CALIBRATION_MIN_SAMPLES].copy()
    if not patterns.empty:
        patterns = patterns.sort_values(["Accuracy", "Matches"], ascending=[True, False])
        with st.expander("Repeated 1X2 weak spots (50+ saved results)"):
            st.caption(
                "These are recurring associations in the saved results, not proof of a cause. "
                "They are shown only after 50 comparable forecasts; self-calibration uses the "
                "same minimum and learns from the model's earlier probabilities and outcomes."
            )
            patterns["Correct / scored"] = patterns.apply(
                lambda row: f"{int(row['Correct'])} / {int(row['Matches'])}", axis=1
            )
            patterns["Accuracy"] = patterns["Accuracy"].map(lambda value: f"{value:.0%}")
            patterns["Mean log loss"] = patterns["MeanLogLoss"].round(3)
            st.dataframe(
                patterns[["LEAGUE", "PREDICTED_OUTCOME", "Confidence band", "Correct / scored", "Accuracy", "Mean log loss"]],
                hide_index=True, use_container_width=True,
            )
    else:
        with st.expander("Repeated 1X2 weak spots"):
            st.info(
                f"No league, predicted-outcome, and confidence group has {CALIBRATION_MIN_SAMPLES} "
                "completed forecasts yet. The app will wait for enough of its own results before "
                "treating a repeated miss as a reliable learning signal."
            )

    recent = feedback.copy()
    recent["Date"] = pd.to_datetime(recent["DATE"], errors="coerce").dt.strftime("%Y-%m-%d")
    recent["Match"] = recent["HOME"].astype(str) + " vs " + recent["AWAY"].astype(str)
    recent["Actual score"] = recent["FTHG"].astype(int).astype(str) + "–" + recent["FTAG"].astype(int).astype(str)
    recent["Outcome check"] = recent.apply(
        lambda row: ("✓ " if row["CORRECT"] else "✗ ") + row["PREDICTED_OUTCOME"] + " / " + row["ACTUAL_OUTCOME"],
        axis=1,
    )
    recent_columns = ["Date", "LEAGUE", "Match", "Outcome check", "Actual score"]
    if "BEST_PICK" in recent and "BEST_PICK_CORRECT" in recent:
        recent["Best pick check"] = recent.apply(
            lambda row: "—" if pd.isna(row["BEST_PICK_CORRECT"]) else
            ("✓ " if bool(row["BEST_PICK_CORRECT"]) else "✗ ") + str(row["BEST_PICK"]), axis=1,
        )
        recent_columns.append("Best pick check")
    for estimate, actual, label in (
        ("EXP_HOME_GOALS", "EXP_AWAY_GOALS", "Expected goals (home–away)"),
    ):
        if {estimate, actual}.issubset(recent.columns):
            recent[label] = recent.apply(
                lambda row: f"{row[estimate]:.1f}–{row[actual]:.1f}" if pd.notna(row[estimate]) and pd.notna(row[actual]) else "—",
                axis=1,
            )
            recent_columns.append(label)
    recent["Total-goal error"] = pd.to_numeric(recent["TOTAL_GOAL_ERROR"], errors="coerce").round(2)
    recent_columns.append("Total-goal error")
    if "ODDS_SOURCE" in recent.columns:
        recent_columns.append("ODDS_SOURCE")
    st.caption(f"Completed forecast record ({len(recent)} matches), newest first")
    st.dataframe(recent[recent_columns], hide_index=True, use_container_width=True)
