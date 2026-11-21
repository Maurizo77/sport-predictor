"""Compare the deployed football model with a boosted Poisson challenger.

Uses a fixed chronological holdout, prints metrics, and does not change project
logs or model files. Run from the project root.
"""

import numpy as np
import pandas as pd
from scipy.stats import poisson
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import accuracy_score, log_loss, mean_absolute_error


TRAIN_BEFORE = "2025-07-01"
TEST_BEFORE = "2026-07-01"
DATA_FILE = "football_games.csv"
STATS = ["GF", "GA", "SF", "SA"]
FEATURES = ["H_" + col for col in STATS] + ["A_" + col for col in STATS] + ["ELO_DIFF"]
LABELS = ["A", "D", "H"]


def build_features():
    matches = pd.read_csv(DATA_FILE)
    matches["Date"] = pd.to_datetime(matches["Date"], dayfirst=True, format="mixed")
    matches = matches.dropna(subset=["FTHG", "FTAG"]).sort_values("Date").reset_index(drop=True)
    matches["MATCH_ID"] = matches.index

    ratings = {}
    elo_diff = []
    for row in matches.itertuples():
        home_rating = ratings.get(row.HomeTeam, 1500)
        away_rating = ratings.get(row.AwayTeam, 1500)
        elo_diff.append((home_rating - away_rating) / 100)
        expected = 1 / (1 + 10 ** ((away_rating - home_rating - 70) / 400))
        result = 1.0 if row.FTHG > row.FTAG else 0.5 if row.FTHG == row.FTAG else 0.0
        margin = abs(row.FTHG - row.FTAG)
        margin_factor = 1.0 if margin <= 1 else 1.5 if margin == 2 else (11 + margin) / 8
        change = 20 * margin_factor * (result - expected)
        ratings[row.HomeTeam] = home_rating + change
        ratings[row.AwayTeam] = away_rating - change
    matches["ELO_DIFF"] = elo_diff

    home = matches[["MATCH_ID", "Date", "HomeTeam", "FTHG", "FTAG", "HST", "AST"]]
    home.columns = ["MATCH_ID", "Date", "TEAM", "GF", "GA", "SF", "SA"]
    away = matches[["MATCH_ID", "Date", "AwayTeam", "FTAG", "FTHG", "AST", "HST"]]
    away.columns = ["MATCH_ID", "Date", "TEAM", "GF", "GA", "SF", "SA"]
    teams = pd.concat([home, away]).sort_values(["TEAM", "Date"])
    for column in STATS:
        teams[column + "_AVG"] = teams.groupby("TEAM")[column].transform(
            lambda values: values.shift(1).rolling(10, min_periods=5).mean()
        )

    form_columns = [column + "_AVG" for column in STATS]
    home_form = teams[["MATCH_ID", "TEAM"] + form_columns].rename(
        columns={"TEAM": "HomeTeam", **{column + "_AVG": "H_" + column for column in STATS}}
    )
    away_form = teams[["MATCH_ID", "TEAM"] + form_columns].rename(
        columns={"TEAM": "AwayTeam", **{column + "_AVG": "A_" + column for column in STATS}}
    )
    matches = matches.merge(home_form, on=["MATCH_ID", "HomeTeam"])
    matches = matches.merge(away_form, on=["MATCH_ID", "AwayTeam"])
    return matches.dropna(subset=FEATURES + ["FTHG", "FTAG", "FTR"])


def outcome_probabilities(home_goals, away_goals):
    goals = np.arange(0, 16)
    grid = poisson.pmf(goals, home_goals[:, None])[:, :, None] * poisson.pmf(
        goals, away_goals[:, None]
    )[:, None, :]
    home_grid, away_grid = goals[:, None], goals[None, :]
    probabilities = np.column_stack([
        (grid * (home_grid < away_grid)).sum(axis=(1, 2)),
        (grid * (home_grid == away_grid)).sum(axis=(1, 2)),
        (grid * (home_grid > away_grid)).sum(axis=(1, 2)),
    ])
    return probabilities / probabilities.sum(axis=1, keepdims=True)


def report(name, test, home_goals, away_goals):
    probabilities = outcome_probabilities(home_goals, away_goals)
    predictions = np.array(LABELS)[probabilities.argmax(axis=1)]
    print(
        f"{name}: home/away goals MAE "
        f"{mean_absolute_error(test['FTHG'], home_goals):.3f}/"
        f"{mean_absolute_error(test['FTAG'], away_goals):.3f}; "
        f"outcome accuracy {accuracy_score(test['FTR'], predictions):.3f}; "
        f"outcome log loss {log_loss(test['FTR'], probabilities, labels=LABELS):.3f}"
    )


data = build_features()
train = data[data["Date"] < TRAIN_BEFORE]
test = data[(data["Date"] >= TRAIN_BEFORE) & (data["Date"] < TEST_BEFORE)].copy()
if train.empty or test.empty:
    raise SystemExit("The selected chronological train/test period has no training or test rows.")

print(f"Training before {TRAIN_BEFORE}; testing {TRAIN_BEFORE} through {TEST_BEFORE}")
print(f"Training games: {len(train)} | test games: {len(test)}")

poisson_home = PoissonRegressor(alpha=0.0001, max_iter=1000).fit(train[FEATURES], train["FTHG"])
poisson_away = PoissonRegressor(alpha=0.0001, max_iter=1000).fit(train[FEATURES], train["FTAG"])
poisson_home_prediction = poisson_home.predict(test[FEATURES])
poisson_away_prediction = poisson_away.predict(test[FEATURES])
report("Current PoissonRegressor", test, poisson_home_prediction, poisson_away_prediction)

boosted_home = HistGradientBoostingRegressor(
    loss="poisson", learning_rate=0.08, max_iter=150, max_leaf_nodes=15,
    l2_regularization=2.0, random_state=42,
).fit(train[FEATURES], train["FTHG"])
boosted_away = HistGradientBoostingRegressor(
    loss="poisson", learning_rate=0.08, max_iter=150, max_leaf_nodes=15,
    l2_regularization=2.0, random_state=42,
).fit(train[FEATURES], train["FTAG"])
boosted_home_prediction = boosted_home.predict(test[FEATURES])
boosted_away_prediction = boosted_away.predict(test[FEATURES])
report("HistGradientBoosting Poisson challenger", test, boosted_home_prediction, boosted_away_prediction)

if {"B365A", "B365D", "B365H"}.issubset(test.columns):
    odds = test[["B365A", "B365D", "B365H"]].to_numpy(dtype=float)
    has_odds = np.isfinite(odds).all(axis=1) & (odds > 0).all(axis=1)
    if has_odds.any():
        implied = 1 / odds[has_odds]
        implied /= implied.sum(axis=1, keepdims=True)
        print(
            f"Bookmaker odds: {has_odds.sum()} matched games; "
            f"outcome accuracy {accuracy_score(test.loc[has_odds, 'FTR'], np.array(LABELS)[implied.argmax(axis=1)]):.3f}; "
            f"outcome log loss {log_loss(test.loc[has_odds, 'FTR'], implied, labels=LABELS):.3f}"
        )
        print("Models on the same bookmaker-covered games:")
        report(
            "  Current PoissonRegressor",
            test.loc[has_odds],
            poisson_home_prediction[has_odds],
            poisson_away_prediction[has_odds],
        )
        report(
            "  HistGradientBoosting challenger",
            test.loc[has_odds],
            boosted_home_prediction[has_odds],
            boosted_away_prediction[has_odds],
        )
