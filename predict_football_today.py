import os
import sys
import tempfile
import time

import numpy as np
import pandas as pd
from scipy.stats import poisson
from sklearn.linear_model import PoissonRegressor

from model_cache import load_model_cache, save_model_cache
from event_time import (
    FOOTBALL_FEED_TIMEZONE,
    TIME_UNAVAILABLE,
    football_event_time_wat,
)

sys.stdout.reconfigure(encoding="utf-8")

BASE = os.path.dirname(os.path.abspath(__file__))
LOG_FILE = os.path.join(BASE, "football_predictions_log.csv")
MANUAL_FILE = "my_fixtures.csv"
FIXTURES_URL = "https://www.football-data.co.uk/fixtures.csv"
DAYS_AHEAD = 14
CACHE_NAME = "football_training.joblib"
CACHE_VERSION = 2

data_path = os.path.join(BASE, "football_games.csv")


def train_pipeline():
    matches = pd.read_csv(data_path)
    matches["Date"] = pd.to_datetime(matches["Date"], dayfirst=True, format="mixed")
    matches = matches.dropna(subset=["FTHG", "FTAG"])
    matches = matches.sort_values("Date").reset_index(drop=True)
    matches["MATCH_ID"] = matches.index

    def margin_factor(goal_diff):
        if goal_diff <= 1:
            return 1.0
        if goal_diff == 2:
            return 1.5
        return (11 + goal_diff) / 8

    ratings = {}
    elo_diff = []
    for row in matches.itertuples():
        rating_home = ratings.get(row.HomeTeam, 1500)
        rating_away = ratings.get(row.AwayTeam, 1500)
        elo_diff.append((rating_home - rating_away) / 100)
        expected = 1 / (1 + 10 ** ((rating_away - rating_home - 70) / 400))
        if row.FTHG > row.FTAG:
            result = 1.0
        elif row.FTHG == row.FTAG:
            result = 0.5
        else:
            result = 0.0
        change = 20 * margin_factor(abs(row.FTHG - row.FTAG)) * (result - expected)
        ratings[row.HomeTeam] = rating_home + change
        ratings[row.AwayTeam] = rating_away - change
    matches["ELO_DIFF"] = elo_diff

    stats = ["GF", "GA", "SF", "SA"]
    home = matches[["MATCH_ID", "Date", "HomeTeam", "FTHG", "FTAG", "HST", "AST"]]
    home.columns = ["MATCH_ID", "Date", "TEAM", "GF", "GA", "SF", "SA"]
    away = matches[["MATCH_ID", "Date", "AwayTeam", "FTAG", "FTHG", "AST", "HST"]]
    away.columns = ["MATCH_ID", "Date", "TEAM", "GF", "GA", "SF", "SA"]

    teams = pd.concat([home, away]).sort_values(["TEAM", "Date"])
    for col in stats:
        teams[col + "_AVG"] = teams.groupby("TEAM")[col].transform(
            lambda series: series.shift(1).rolling(10, min_periods=5).mean()
        )

    form_cols = [col + "_AVG" for col in stats]
    home_form = teams[["MATCH_ID", "TEAM"] + form_cols].rename(
        columns={"TEAM": "HomeTeam", **{col + "_AVG": "H_" + col for col in stats}}
    )
    away_form = teams[["MATCH_ID", "TEAM"] + form_cols].rename(
        columns={"TEAM": "AwayTeam", **{col + "_AVG": "A_" + col for col in stats}}
    )
    matches = matches.merge(home_form, on=["MATCH_ID", "HomeTeam"])
    matches = matches.merge(away_form, on=["MATCH_ID", "AwayTeam"])

    features = ["H_" + col for col in stats] + ["A_" + col for col in stats] + ["ELO_DIFF"]
    train = matches.dropna(subset=features + ["FTHG", "FTAG"])
    home_model = PoissonRegressor(alpha=0.0001, max_iter=1000)
    home_model.fit(train[features], train["FTHG"])
    away_model = PoissonRegressor(alpha=0.0001, max_iter=1000)
    away_model.fit(train[features], train["FTAG"])

    recent = teams.groupby("TEAM").tail(10)
    latest_form = recent.groupby("TEAM")[stats].mean()
    latest_form = latest_form[recent.groupby("TEAM").size() >= 5]
    return {
        "history_end": matches["Date"].max().date().isoformat(),
        "league_codes": matches["LEAGUE"].unique().tolist(),
        "stats": stats,
        "features": features,
        "ratings": ratings,
        "latest_form": latest_form,
        "home_model": home_model,
        "away_model": away_model,
    }


pipeline_started = time.perf_counter()
trained = load_model_cache(CACHE_NAME, data_path, CACHE_VERSION)
if trained is None:
    print("Preparing and training football models from match history...")
    trained = train_pipeline()
    save_model_cache(CACHE_NAME, data_path, CACHE_VERSION, trained)
else:
    print("Loaded cached football training pipeline; match history has not changed.")
print("Training pipeline time:", round(time.perf_counter() - pipeline_started, 2), "seconds")

stats = trained["stats"]
features = trained["features"]
ratings = trained["ratings"]
latest_form = trained["latest_form"]
home_model = trained["home_model"]
away_model = trained["away_model"]

print("History up to:", trained["history_end"])

source = MANUAL_FILE if os.path.exists(MANUAL_FILE) else FIXTURES_URL
print("Fixtures from:", source)
try:
    fixtures = pd.read_csv(
        source, encoding="utf-8-sig", encoding_errors="replace", on_bad_lines="skip"
    )
except Exception as error:
    print("Could not read the fixtures:", type(error).__name__)
    raise SystemExit

needed = ["Div", "Date", "HomeTeam", "AwayTeam"]
missing = [c for c in needed if c not in fixtures.columns]
if missing:
    print("The fixtures file has no column(s):", missing)
    print("Columns found:", list(fixtures.columns)[:12])
    raise SystemExit

fixtures = fixtures.dropna(subset=["HomeTeam", "AwayTeam"])
if "Time" not in fixtures.columns:
    fixtures["Time"] = None
fixtures["Date"] = pd.to_datetime(
    fixtures["Date"], dayfirst=True, format="mixed", errors="coerce"
)
today = pd.Timestamp.now(tz=FOOTBALL_FEED_TIMEZONE).normalize().tz_localize(None)
fixtures = fixtures[
    (fixtures["Date"] >= today)
    & (fixtures["Date"] <= today + pd.Timedelta(days=DAYS_AHEAD))
    & (fixtures["Div"].isin(trained["league_codes"]))
]
if fixtures.empty:
    print("No fixtures in the next", DAYS_AHEAD, "days for our leagues.")
    raise SystemExit

rows = []
for fx in fixtures.itertuples():
    if fx.HomeTeam not in latest_form.index or fx.AwayTeam not in latest_form.index:
        print("Skipping", fx.HomeTeam, "vs", fx.AwayTeam, "(team name not found in our data)")
        continue
    row = {
        "DATE": fx.Date.date(),
        "LEAGUE": fx.Div,
        "HOME": fx.HomeTeam,
        "AWAY": fx.AwayTeam,
        "EVENT_TIME_WAT": football_event_time_wat(
            fx.Date,
            fx.Time,
            getattr(fx, "TimeZone", FOOTBALL_FEED_TIMEZONE),
        ),
    }
    for c in stats:
        row["H_" + c] = latest_form.loc[fx.HomeTeam, c]
        row["A_" + c] = latest_form.loc[fx.AwayTeam, c]
    row["ELO_DIFF"] = (ratings.get(fx.HomeTeam, 1500) - ratings.get(fx.AwayTeam, 1500)) / 100
    rows.append(row)

X = pd.DataFrame(rows).dropna(subset=features).reset_index(drop=True)
if X.empty:
    print("No matches could be predicted.")
    raise SystemExit

lam_home = home_model.predict(X[features])
lam_away = away_model.predict(X[features])

goals = np.arange(0, 11)
grid = (
    poisson.pmf(goals, lam_home[:, None])[:, :, None]
    * poisson.pmf(goals, lam_away[:, None])[:, None, :]
)
i = goals[:, None]
j = goals[None, :]

p_home = (grid * (i > j)).sum(axis=(1, 2))
p_draw = (grid * (i == j)).sum(axis=(1, 2))
p_away = (grid * (i < j)).sum(axis=(1, 2))
norm = p_home + p_draw + p_away
p_home, p_draw, p_away = p_home / norm, p_draw / norm, p_away / norm
p_o15 = (grid * (i + j > 1.5)).sum(axis=(1, 2))
p_o25 = (grid * (i + j > 2.5)).sum(axis=(1, 2))
p_o35 = (grid * (i + j > 3.5)).sum(axis=(1, 2))
p_btts = (grid * ((i > 0) & (j > 0))).sum(axis=(1, 2))


def pct(x):
    return str(round(x * 100)) + "%"


picks = []

print("Predictions for matches in the next", DAYS_AHEAD, "days")
for k in range(len(X)):
    r = X.iloc[k]
    game = r["HOME"] + " vs " + r["AWAY"]
    flat = grid[k].flatten()
    best = flat.argsort()[::-1][:3]
    top_scores = ", ".join(str(b // 11) + "-" + str(b % 11) + " " + pct(flat[b]) for b in best)
    print()
    print(r["LEAGUE"], r["DATE"], "| kickoff", r["EVENT_TIME_WAT"], "|", game)
    print("  Win/draw/loss: " + pct(p_home[k]) + " / " + pct(p_draw[k]) + " / " + pct(p_away[k]))
    print("  Double chance 1X / X2 / 12: " + pct(p_home[k] + p_draw[k]) + " / "
          + pct(p_draw[k] + p_away[k]) + " / " + pct(p_home[k] + p_away[k]))
    print("  Over 1.5 / 2.5 / 3.5 goals: " + pct(p_o15[k]) + " / " + pct(p_o25[k]) + " / " + pct(p_o35[k]))
    print("  Both teams score: " + pct(p_btts[k]))
    print("  Likely scores: " + top_scores)

    options = {
        "Home win": p_home[k], "Draw": p_draw[k], "Away win": p_away[k],
        "Home or draw": p_home[k] + p_draw[k],
        "Draw or away": p_draw[k] + p_away[k],
        "Home or away": p_home[k] + p_away[k],
        "Over 1.5": p_o15[k], "Under 1.5": 1 - p_o15[k],
        "Over 2.5": p_o25[k], "Under 2.5": 1 - p_o25[k],
        "Over 3.5": p_o35[k], "Under 3.5": 1 - p_o35[k],
        "BTTS yes": p_btts[k], "BTTS no": 1 - p_btts[k],
    }
    # Keep one highest-probability option per match so the shortlist cannot
    # be filled with several overlapping markets from the same fixture.
    best_name = max(options, key=options.get)
    picks.append((options[best_name], game, best_name))

picks.sort(reverse=True)
print()
print("Most confident options (higher chance means lower odds):")
shortlist = picks[:12]
for prob, game, name in shortlist:
    print(" ", pct(prob), game, "->", name, "| fair odds", round(1 / prob, 2))

best_by_game = {game: (name, prob) for prob, game, name in picks}
bet_picks = [best_by_game[r["HOME"] + " vs " + r["AWAY"]][0] for _, r in X.iterrows()]
bet_probs = [best_by_game[r["HOME"] + " vs " + r["AWAY"]][1] for _, r in X.iterrows()]

log = pd.DataFrame({
    "DATE": X["DATE"], "EVENT_TIME_WAT": X["EVENT_TIME_WAT"],
    "LEAGUE": X["LEAGUE"], "HOME": X["HOME"], "AWAY": X["AWAY"],
    "P_HOME": p_home.round(3), "P_DRAW": p_draw.round(3), "P_AWAY": p_away.round(3),
    "P_OVER25": p_o25.round(3), "P_BTTS": p_btts.round(3),
    "EXP_HOME_GOALS": lam_home.round(2), "EXP_AWAY_GOALS": lam_away.round(2),
    "BEST_PICK": bet_picks, "BEST_PICK_PROB": np.round(bet_probs, 3),
})
if os.path.exists(LOG_FILE):
    old = pd.read_csv(LOG_FILE)
    old_keys = old["DATE"].astype(str) + old["HOME"] + old["AWAY"]
    new_keys = log["DATE"].astype(str) + log["HOME"] + log["AWAY"]
    # Refresh matching fixtures so new shortlist fields are recorded and
    # reruns update predictions for fixtures that have not started yet.
    old = old[~old_keys.isin(new_keys)]
    if "EVENT_TIME_WAT" not in old:
        old["EVENT_TIME_WAT"] = TIME_UNAVAILABLE
    columns = list(dict.fromkeys(list(old.columns) + list(log.columns)))
    combined = pd.concat([old, log], ignore_index=True).reindex(columns=columns)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="", dir=BASE, suffix=".tmp", delete=False
    ) as temporary:
        temporary_path = temporary.name
    try:
        combined.to_csv(temporary_path, index=False)
        os.replace(temporary_path, LOG_FILE)
    finally:
        if os.path.exists(temporary_path):
            os.remove(temporary_path)
else:
    log.to_csv(LOG_FILE, index=False)
print()
print("Saved or refreshed", len(log), "predictions to", LOG_FILE)
