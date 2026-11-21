"""Join saved football forecasts to completed match results for honest review."""

import os
import tempfile

import numpy as np
import pandas as pd


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

    merged = prediction_data.merge(
        actual_data[key + ["FTHG", "FTAG"]], on=key, how="inner", validate="one_to_one"
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
