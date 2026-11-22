"""Join saved football forecasts to completed match results for honest review."""

import os
import tempfile

import numpy as np
import pandas as pd
from scipy.stats import poisson

from football_market_features import FEATURE_GROUPS


PREDICTIONS_FILE = "football_predictions_log.csv"
GAMES_FILE = "football_games.csv"
FEEDBACK_FILE = "football_feedback_log.csv"


def build_feedback(base):
    prediction_path = os.path.join(base, PREDICTIONS_FILE)
    games_path = os.path.join(base, GAMES_FILE)
    if not os.path.exists(prediction_path) or not os.path.exists(games_path):
        return pd.DataFrame()

    predictions = pd.read_csv(prediction_path)
    games = pd.read_csv(games_path)
    prediction_columns = {
        "DATE", "LEAGUE", "HOME", "AWAY", "P_HOME", "P_DRAW", "P_AWAY",
    }
    actual_columns = {"Date", "LEAGUE", "HomeTeam", "AwayTeam", "FTHG", "FTAG"}
    if not prediction_columns.issubset(predictions.columns):
        return pd.DataFrame()
    if not actual_columns.issubset(games.columns):
        return pd.DataFrame()

    predictions["DATE"] = pd.to_datetime(predictions["DATE"], errors="coerce")
    games["DATE"] = pd.to_datetime(
        games["Date"], dayfirst=True, format="mixed", errors="coerce"
    )
    for column in ("P_HOME", "P_DRAW", "P_AWAY"):
        predictions[column] = pd.to_numeric(predictions[column], errors="coerce")
    games["FTHG"] = pd.to_numeric(games["FTHG"], errors="coerce")
    games["FTAG"] = pd.to_numeric(games["FTAG"], errors="coerce")

    key = ["DATE", "LEAGUE", "HOME", "AWAY"]
    prediction_data = predictions.dropna(
        subset=key + ["P_HOME", "P_DRAW", "P_AWAY"]
    ).drop_duplicates(key, keep="first")
    actual_data = games.dropna(
        subset=["DATE", "LEAGUE", "HomeTeam", "AwayTeam", "FTHG", "FTAG"]
    ).rename(columns={"HomeTeam": "HOME", "AwayTeam": "AWAY"})
    actual_data = actual_data.drop_duplicates(key, keep="last")

    actual_stat_columns = [
        "FTHG", "FTAG", "HTHG", "HTAG", "HC", "AC", "HS", "AS", "HST", "AST",
        "HF", "AF", "HO", "AO", "HY", "AY", "HR", "AR",
    ]
    actual_stat_columns = [column for column in actual_stat_columns if column in actual_data]
    for column in actual_stat_columns:
        actual_data[column] = pd.to_numeric(actual_data[column], errors="coerce")
    merged = prediction_data.merge(
        actual_data[key + actual_stat_columns], on=key, how="inner", validate="one_to_one"
    )
    if merged.empty:
        return merged

    probabilities = merged[["P_HOME", "P_DRAW", "P_AWAY"]].to_numpy(dtype=float)
    probabilities = np.clip(probabilities, 1e-6, 1)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    merged[["P_HOME", "P_DRAW", "P_AWAY"]] = probabilities
    actual = np.where(
        merged["FTHG"].to_numpy() > merged["FTAG"].to_numpy(), "Home win",
        np.where(merged["FTHG"].to_numpy() == merged["FTAG"].to_numpy(), "Draw", "Away win"),
    )
    outcome_names = np.array(["Home win", "Draw", "Away win"])
    predicted = outcome_names[probabilities.argmax(axis=1)]
    actual_index = np.where(actual == "Home win", 0, np.where(actual == "Draw", 1, 2))
    merged["PREDICTED_OUTCOME"] = predicted
    merged["ACTUAL_OUTCOME"] = actual
    merged["CORRECT"] = predicted == actual
    merged["LOG_LOSS"] = -np.log(probabilities[np.arange(len(merged)), actual_index])
    actual_one_hot = np.eye(3)[actual_index]
    merged["BRIER_SCORE"] = ((probabilities - actual_one_hot) ** 2).sum(axis=1)
    merged["HOME_GOAL_ERROR"] = (merged["EXP_HOME_GOALS"] - merged["FTHG"]).abs() if "EXP_HOME_GOALS" in merged else np.nan
    merged["AWAY_GOAL_ERROR"] = (merged["EXP_AWAY_GOALS"] - merged["FTAG"]).abs() if "EXP_AWAY_GOALS" in merged else np.nan
    merged["TOTAL_GOAL_ERROR"] = (
        (merged["EXP_HOME_GOALS"] + merged["EXP_AWAY_GOALS"] - merged["FTHG"] - merged["FTAG"]).abs()
        if {"EXP_HOME_GOALS", "EXP_AWAY_GOALS"}.issubset(merged.columns) else np.nan
    )

    probability_fallbacks = {}

    def add_probability_fallback(column, fallback):
        fallback = pd.Series(pd.to_numeric(fallback, errors="coerce"), index=merged.index)
        if column in merged:
            merged[column] = pd.to_numeric(merged[column], errors="coerce").fillna(fallback)
        else:
            probability_fallbacks[column] = fallback

    add_probability_fallback("P_HOME_OR_DRAW", merged["P_HOME"] + merged["P_DRAW"])
    add_probability_fallback("P_DRAW_OR_AWAY", merged["P_DRAW"] + merged["P_AWAY"])
    add_probability_fallback("P_HOME_OR_AWAY", merged["P_HOME"] + merged["P_AWAY"])
    if {"EXP_HOME_GOALS", "EXP_AWAY_GOALS"}.issubset(merged.columns):
        total_goal_rate = (
            pd.to_numeric(merged["EXP_HOME_GOALS"], errors="coerce")
            + pd.to_numeric(merged["EXP_AWAY_GOALS"], errors="coerce")
        )
        add_probability_fallback("P_OVER15", poisson.sf(1, total_goal_rate))
        add_probability_fallback("P_OVER35", poisson.sf(3, total_goal_rate))
    for target, source in {
        "P_UNDER15": "P_OVER15", "P_UNDER25": "P_OVER25", "P_UNDER35": "P_OVER35",
        "P_BTTS_NO": "P_BTTS", "P_FIRST_HALF_UNDER15": "P_FIRST_HALF_2PLUS",
        "P_NO_FIRST_HALF_GOAL": "P_FIRST_HALF_GOAL", "P_UNDER85_CORNERS": "P_OVER85_CORNERS",
        "P_UNDER215_SHOTS": "P_OVER215_SHOTS", "P_UNDER75_SHOTS_ON_TARGET": "P_OVER75_SHOTS_ON_TARGET",
        "P_UNDER35_YELLOW_CARDS": "P_OVER35_YELLOW_CARDS", "P_UNDER215_FOULS": "P_OVER215_FOULS",
        "P_UNDER25_OFFSIDES": "P_OVER25_OFFSIDES", "P_NO_RED_CARD": "P_ANY_RED_CARD",
    }.items():
        if source in probability_fallbacks:
            source_probability = probability_fallbacks[source]
        elif source in merged:
            source_probability = pd.to_numeric(merged[source], errors="coerce")
        else:
            continue
        add_probability_fallback(target, 1 - source_probability)
    if probability_fallbacks:
        merged = pd.concat([merged, pd.DataFrame(probability_fallbacks, index=merged.index)], axis=1)

    # Score the model's headline pick across outcomes and match-stat markets.
    # A pick is marked unavailable when the historical result lacks that stat.
    if "BEST_PICK" in merged:
        actual_market_values = {}
        if {"FTHG", "FTAG"}.issubset(merged.columns):
            total_goals = merged["FTHG"] + merged["FTAG"]
            actual_market_values.update({
                "home win": merged["FTHG"] > merged["FTAG"],
                "draw": merged["FTHG"] == merged["FTAG"],
                "away win": merged["FTHG"] < merged["FTAG"],
                "home or draw": merged["FTHG"] >= merged["FTAG"],
                "draw or away": merged["FTHG"] <= merged["FTAG"],
                "home or away": merged["FTHG"] != merged["FTAG"],
                "home win or draw (1x)": merged["FTHG"] >= merged["FTAG"],
                "draw or away win (x2)": merged["FTHG"] <= merged["FTAG"],
                "either team to win (12)": merged["FTHG"] != merged["FTAG"],
                "over 1.5": total_goals >= 2,
                "under 1.5": total_goals <= 1,
                "over 2.5": total_goals >= 3,
                "under 2.5": total_goals <= 2,
                "over 3.5": total_goals >= 4,
                "under 3.5": total_goals <= 3,
                "btts yes": (merged["FTHG"] >= 1) & (merged["FTAG"] >= 1),
                "btts no": (merged["FTHG"] == 0) | (merged["FTAG"] == 0),
            })
        if {"HTHG", "HTAG"}.issubset(merged.columns):
            half_total = merged["HTHG"] + merged["HTAG"]
            actual_market_values.update({
                "first-half home win": (merged["HTHG"] > merged["HTAG"]).where(merged[["HTHG", "HTAG"]].notna().all(axis=1)),
                "first-half draw": (merged["HTHG"] == merged["HTAG"]).where(merged[["HTHG", "HTAG"]].notna().all(axis=1)),
                "first-half away win": (merged["HTHG"] < merged["HTAG"]).where(merged[["HTHG", "HTAG"]].notna().all(axis=1)),
                "first-half goal": (half_total >= 1).where(half_total.notna()),
                "no first-half goal": (half_total == 0).where(half_total.notna()),
                "2+ first-half goals": (half_total >= 2).where(half_total.notna()),
                "under 1.5 first-half goals": (half_total <= 1).where(half_total.notna()),
            })
        count_markets = {
            "corners": ("HC", "AC", 9), "shots": ("HS", "AS", 22),
            "shots on target": ("HST", "AST", 8), "yellow cards": ("HY", "AY", 4),
            "fouls": ("HF", "AF", 22), "offsides": ("HO", "AO", 3),
            "red card": ("HR", "AR", 1),
        }
        for label, (home_col, away_col, threshold) in count_markets.items():
            if {home_col, away_col}.issubset(merged.columns):
                total = merged[home_col] + merged[away_col]
                available = merged[[home_col, away_col]].notna().all(axis=1)
                actual_market_values[f"over {threshold - 0.5} {label}"] = (total >= threshold).where(available)
                actual_market_values[f"under {threshold - 0.5} {label}"] = (total < threshold).where(available)
                if label == "red card":
                    actual_market_values["red card"] = (total >= threshold).where(available)
                    actual_market_values["no red card"] = (total < threshold).where(available)

        pick_actual = pd.Series(pd.NA, index=merged.index, dtype="boolean")
        for pick, actual_value in actual_market_values.items():
            mask = merged["BEST_PICK"].fillna("").astype(str).str.strip().str.casefold().eq(pick)
            pick_actual.loc[mask] = actual_value.loc[mask].astype("boolean")
        merged["BEST_PICK_CORRECT"] = pick_actual

    def event_series(mask, required_columns):
        if not set(required_columns).issubset(merged.columns):
            return None
        available = merged[list(required_columns)].notna().all(axis=1)
        return pd.Series(mask, index=merged.index).astype("boolean").where(available)

    actual_events = {}
    full_total = merged["FTHG"] + merged["FTAG"]
    actual_events.update({
        "HOME_WIN": event_series(merged["FTHG"] > merged["FTAG"], ("FTHG", "FTAG")),
        "DRAW": event_series(merged["FTHG"] == merged["FTAG"], ("FTHG", "FTAG")),
        "AWAY_WIN": event_series(merged["FTHG"] < merged["FTAG"], ("FTHG", "FTAG")),
        "HOME_DRAW": event_series(merged["FTHG"] >= merged["FTAG"], ("FTHG", "FTAG")),
        "DRAW_AWAY": event_series(merged["FTHG"] <= merged["FTAG"], ("FTHG", "FTAG")),
        "HOME_AWAY": event_series(merged["FTHG"] != merged["FTAG"], ("FTHG", "FTAG")),
        "OVER15": event_series(full_total >= 2, ("FTHG", "FTAG")),
        "UNDER15": event_series(full_total <= 1, ("FTHG", "FTAG")),
        "OVER25": event_series(full_total >= 3, ("FTHG", "FTAG")),
        "UNDER25": event_series(full_total <= 2, ("FTHG", "FTAG")),
        "OVER35": event_series(full_total >= 4, ("FTHG", "FTAG")),
        "UNDER35": event_series(full_total <= 3, ("FTHG", "FTAG")),
        "BTTS_YES": event_series((merged["FTHG"] >= 1) & (merged["FTAG"] >= 1), ("FTHG", "FTAG")),
        "BTTS_NO": event_series((merged["FTHG"] == 0) | (merged["FTAG"] == 0), ("FTHG", "FTAG")),
    })
    if {"HTHG", "HTAG"}.issubset(merged.columns):
        half_total = merged["HTHG"] + merged["HTAG"]
        actual_events.update({
            "HT_HOME": event_series(merged["HTHG"] > merged["HTAG"], ("HTHG", "HTAG")),
            "HT_DRAW": event_series(merged["HTHG"] == merged["HTAG"], ("HTHG", "HTAG")),
            "HT_AWAY": event_series(merged["HTHG"] < merged["HTAG"], ("HTHG", "HTAG")),
            "HT_GOAL": event_series(half_total >= 1, ("HTHG", "HTAG")),
            "HT_NO_GOAL": event_series(half_total == 0, ("HTHG", "HTAG")),
            "HT_2PLUS": event_series(half_total >= 2, ("HTHG", "HTAG")),
            "HT_UNDER15": event_series(half_total <= 1, ("HTHG", "HTAG")),
        })
    count_event_specs = {
        "CORNERS": ("HC", "AC", 9), "SHOTS": ("HS", "AS", 22),
        "SOT": ("HST", "AST", 8), "YELLOW": ("HY", "AY", 4),
        "RED": ("HR", "AR", 1), "FOULS": ("HF", "AF", 22),
        "OFFSIDES": ("HO", "AO", 3),
    }
    for prefix, (home_stat, away_stat, cutoff) in count_event_specs.items():
        if {home_stat, away_stat}.issubset(merged.columns):
            total = merged[home_stat] + merged[away_stat]
            required = (home_stat, away_stat)
            actual_events[prefix + "_OVER"] = event_series(total >= cutoff, required)
            actual_events[prefix + "_UNDER"] = event_series(total < cutoff, required)
    for feature_key, source_key in {
        "CORNERS_OVER85": "CORNERS_OVER", "CORNERS_UNDER85": "CORNERS_UNDER",
        "SHOTS_OVER215": "SHOTS_OVER", "SHOTS_UNDER215": "SHOTS_UNDER",
        "SOT_OVER75": "SOT_OVER", "SOT_UNDER75": "SOT_UNDER",
        "YELLOW_OVER35": "YELLOW_OVER", "YELLOW_UNDER35": "YELLOW_UNDER",
        "RED_YES": "RED_OVER", "RED_NO": "RED_UNDER",
        "FOULS_OVER215": "FOULS_OVER", "FOULS_UNDER215": "FOULS_UNDER",
        "OFFSIDES_OVER25": "OFFSIDES_OVER", "OFFSIDES_UNDER25": "OFFSIDES_UNDER",
    }.items():
        if source_key in actual_events:
            actual_events[feature_key] = actual_events[source_key]
    feature_feedback_columns = {}
    for group_markets in FEATURE_GROUPS.values():
        for feature_key, _label, probability_column, outcome_key in group_markets:
            if probability_column not in merged or outcome_key not in actual_events:
                continue
            probability = pd.to_numeric(merged[probability_column], errors="coerce")
            feature_feedback_columns["FEATURE_PROB_" + feature_key] = probability
            feature_feedback_columns["FEATURE_HIT_" + feature_key] = actual_events[outcome_key].where(
                probability.notna()
            )
    if feature_feedback_columns:
        merged = pd.concat(
            [merged, pd.DataFrame(feature_feedback_columns, index=merged.index)], axis=1
        )

    stat_estimates = {
        "EXP_HOME_CORNERS": "HC", "EXP_AWAY_CORNERS": "AC",
        "EXP_HOME_SHOTS": "HS", "EXP_AWAY_SHOTS": "AS",
        "EXP_HOME_SHOTS_ON_TARGET": "HST", "EXP_AWAY_SHOTS_ON_TARGET": "AST",
        "EXP_HOME_FOULS": "HF", "EXP_AWAY_FOULS": "AF",
        "EXP_HOME_OFFSIDES": "HO", "EXP_AWAY_OFFSIDES": "AO",
        "EXP_HOME_HT_GOALS": "HTHG", "EXP_AWAY_HT_GOALS": "HTAG",
        "EXP_HOME_YELLOW_CARDS": "HY", "EXP_AWAY_YELLOW_CARDS": "AY",
        "EXP_HOME_RED_CARDS": "HR", "EXP_AWAY_RED_CARDS": "AR",
    }
    error_columns = {}
    for estimate, actual_column in stat_estimates.items():
        if estimate in merged and actual_column in merged:
            error_columns["ERROR_" + estimate] = (
                pd.to_numeric(merged[estimate], errors="coerce")
                - pd.to_numeric(merged[actual_column], errors="coerce")
            ).abs()
    if error_columns:
        merged = pd.concat([merged, pd.DataFrame(error_columns, index=merged.index)], axis=1)
    if "ODDS_SOURCE" not in merged:
        merged["ODDS_SOURCE"] = "Model"
    return merged.sort_values("DATE", ascending=False).reset_index(drop=True)


def update_feedback_file(base):
    """Refresh the compact outcomes file used by the phone dashboard publisher."""
    feedback = build_feedback(base)
    if feedback.empty:
        return 0

    output_path = os.path.join(base, FEEDBACK_FILE)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="", dir=base, suffix=".tmp", delete=False
    ) as temporary:
        temporary_path = temporary.name
    try:
        feedback.to_csv(temporary_path, index=False)
        os.replace(temporary_path, output_path)
    finally:
        if os.path.exists(temporary_path):
            os.remove(temporary_path)
    return len(feedback)
