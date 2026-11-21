import difflib
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import poisson
from sklearn.linear_model import PoissonRegressor

sys.stdout.reconfigure(encoding="utf-8")
pd.set_option("display.width", 220)

POOL_FILE = "pool_fixtures.csv"
OUT_FILE = "pool_picks.csv"
FIXTURES_URL = "https://www.football-data.co.uk/fixtures.csv"
DRAW_PICKS = 2
SCORE_PICKS = 2
ALLOW_OVERLAP = True

if not os.path.exists(POOL_FILE):
    print("Create", POOL_FILE, "first, with the columns HomeTeam,AwayTeam (one fixture per row).")
    raise SystemExit

matches = pd.read_csv("football_games.csv")
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

form_sources = {
    "GF": ("FTHG", "FTAG"), "GA": ("FTAG", "FTHG"),
    "SF": ("HST", "AST"), "SA": ("AST", "HST"),
    "SH": ("HS", "AS"), "SHA": ("AS", "HS"),
    "CF": ("HC", "AC"), "CA": ("AC", "HC"),
    "YF": ("HY", "AY"), "YA": ("AY", "HY"),
    "RF": ("HR", "AR"), "RA": ("AR", "HR"),
    "HTF": ("HTHG", "HTAG"), "HTA": ("HTAG", "HTHG"),
    "FOULS_FOR": ("HF", "AF"), "FOULS_AGAINST": ("AF", "HF"),
    "OFFSIDES_FOR": ("HO", "AO"), "OFFSIDES_AGAINST": ("AO", "HO"),
}
stats = []
for stat, (home_source, away_source) in list(form_sources.items()):
    if home_source in matches and away_source in matches:
        home_values = pd.to_numeric(matches[home_source], errors="coerce")
        away_values = pd.to_numeric(matches[away_source], errors="coerce")
        if min(home_values.notna().sum(), away_values.notna().sum()) >= 500:
            stats.append(stat)
            form_sources[stat] = (home_values, away_values)

home = matches[["MATCH_ID", "Date", "HomeTeam"]].rename(columns={"HomeTeam": "TEAM"}).copy()
away = matches[["MATCH_ID", "Date", "AwayTeam"]].rename(columns={"AwayTeam": "TEAM"}).copy()
for stat in stats:
    home[stat] = form_sources[stat][0].to_numpy()
    away[stat] = form_sources[stat][1].to_numpy()

teams = pd.concat([home, away]).sort_values(["TEAM", "Date"])
for col in stats:
    teams[col + "_AVG"] = teams.groupby("TEAM")[col].transform(
        lambda s: s.shift(1).rolling(10, min_periods=5).mean()
    )

form_cols = [c + "_AVG" for c in stats]
home_form = teams[["MATCH_ID", "TEAM"] + form_cols].rename(
    columns={"TEAM": "HomeTeam", **{c + "_AVG": "H_" + c for c in stats}}
)
away_form = teams[["MATCH_ID", "TEAM"] + form_cols].rename(
    columns={"TEAM": "AwayTeam", **{c + "_AVG": "A_" + c for c in stats}}
)
matches = matches.merge(home_form, on=["MATCH_ID", "HomeTeam"])
matches = matches.merge(away_form, on=["MATCH_ID", "AwayTeam"])

full_features = ["H_" + c for c in stats] + ["A_" + c for c in stats] + ["ELO_DIFF"]
basic_features = ["H_GF", "H_GA", "A_GF", "A_GA", "ELO_DIFF"]

train_full = matches.dropna(subset=full_features + ["FTHG", "FTAG"])
train_basic = matches.dropna(subset=basic_features + ["FTHG", "FTAG"])
models = {}
for name, data, feats in (("full", train_full, full_features), ("basic", train_basic, basic_features)):
    models[name] = (
        PoissonRegressor(alpha=0.0001, max_iter=1000).fit(data[feats], data["FTHG"]),
        PoissonRegressor(alpha=0.0001, max_iter=1000).fit(data[feats], data["FTAG"]),
    )

recent = teams.groupby("TEAM").tail(10)
latest_form = recent.groupby("TEAM")[stats].mean()
latest_form = latest_form[recent.groupby("TEAM").size() >= 5]
team_league = matches.groupby("HomeTeam")["LEAGUE"].last()

print("History up to:", matches["Date"].max().date())

pool = pd.read_csv(POOL_FILE, encoding="utf-8-sig", encoding_errors="replace")
pool.columns = [c.strip() for c in pool.columns]
for needed in ("HomeTeam", "AwayTeam"):
    if needed not in pool.columns:
        print("The pool file needs a column called", needed)
        raise SystemExit
pool = pool.dropna(subset=["HomeTeam", "AwayTeam"])
pool["HomeTeam"] = pool["HomeTeam"].astype(str).str.strip()
pool["AwayTeam"] = pool["AwayTeam"].astype(str).str.strip()
print("Fixtures in the pool file:", len(pool))

known = list(latest_form.index)
unmatched = sorted({t for t in list(pool["HomeTeam"]) + list(pool["AwayTeam"]) if t not in known})
if unmatched:
    print()
    print("These names were not found in our data (fix the spelling in the pool file):")
    for name in unmatched:
        close = difflib.get_close_matches(name, known, n=3, cutoff=0.5)
        print("  ", name, "-> did you mean:", ", ".join(close) if close else "no close match")

rows = []
for fx in pool.itertuples():
    if fx.HomeTeam not in known or fx.AwayTeam not in known:
        continue
    row = {"HomeTeam": fx.HomeTeam, "AwayTeam": fx.AwayTeam}
    for c in stats:
        row["H_" + c] = latest_form.loc[fx.HomeTeam, c]
        row["A_" + c] = latest_form.loc[fx.AwayTeam, c]
    row["ELO_DIFF"] = (ratings.get(fx.HomeTeam, 1500) - ratings.get(fx.AwayTeam, 1500)) / 100
    for extra in ("OddsHome", "OddsDraw", "OddsAway"):
        row[extra] = getattr(fx, extra, np.nan)
    rows.append(row)

if not rows:
    print("No fixture could be rated.")
    raise SystemExit

X = pd.DataFrame(rows)
rated = len(X)

lam_home = pd.Series(np.nan, index=X.index)
lam_away = pd.Series(np.nan, index=X.index)
use_full = X[full_features].notna().all(axis=1)
use_basic = ~use_full & X[basic_features].notna().all(axis=1)
if use_full.any():
    lam_home[use_full] = models["full"][0].predict(X.loc[use_full, full_features])
    lam_away[use_full] = models["full"][1].predict(X.loc[use_full, full_features])
if use_basic.any():
    lam_home[use_basic] = models["basic"][0].predict(X.loc[use_basic, basic_features])
    lam_away[use_basic] = models["basic"][1].predict(X.loc[use_basic, basic_features])

keep = lam_home.notna()
X = X[keep].reset_index(drop=True)
lam_home = lam_home[keep].reset_index(drop=True)
lam_away = lam_away[keep].reset_index(drop=True)
if X.empty:
    print("No fixture had enough data for a prediction.")
    raise SystemExit

goals = np.arange(0, 11)
grid = (
    poisson.pmf(goals, lam_home.values[:, None])[:, :, None]
    * poisson.pmf(goals, lam_away.values[:, None])[:, None, :]
)
i = goals[:, None]
j = goals[None, :]
p_home = (grid * (i > j)).sum(axis=(1, 2))
p_draw = (grid * (i == j)).sum(axis=(1, 2))
p_away = (grid * (i < j)).sum(axis=(1, 2))
X["MODEL"] = p_draw / (p_home + p_draw + p_away)

flat = grid.reshape(len(X), -1)
top_index = flat.argmax(axis=1)
X["TOP_SCORE"] = [str(k // 11) + "-" + str(k % 11) for k in top_index]
X["TOP_SCORE_PROB"] = flat.max(axis=1)
draw_cells = np.stack([grid[:, k, k] for k in range(11)], axis=1)
best_draw = draw_cells.argmax(axis=1)
X["DRAW_SCORE"] = [str(k) + "-" + str(k) for k in best_draw]
X["DRAW_SCORE_PROB"] = draw_cells.max(axis=1)
top_three = []
for row_probs in flat:
    order = row_probs.argsort()[::-1][:3]
    top_three.append(", ".join(str(k // 11) + "-" + str(k % 11) + " " + str(round(row_probs[k] * 100)) + "%" for k in order))
X["LIKELY_SCORES"] = top_three

odds = None
try:
    feed = pd.read_csv(FIXTURES_URL, encoding="utf-8-sig", encoding_errors="replace", on_bad_lines="skip")
    needed = ["HomeTeam", "AwayTeam", "B365H", "B365D", "B365A"]
    if all(c in feed.columns for c in needed):
        odds = feed[needed].drop_duplicates(["HomeTeam", "AwayTeam"])
except Exception as error:
    print("Could not download Bet365 prices:", type(error).__name__)

X["H_ODDS"] = np.nan
X["D_ODDS"] = np.nan
X["A_ODDS"] = np.nan
if odds is not None:
    X = X.merge(odds, on=["HomeTeam", "AwayTeam"], how="left")
    X["H_ODDS"], X["D_ODDS"], X["A_ODDS"] = X["B365H"], X["B365D"], X["B365A"]
for source, target in (("OddsHome", "H_ODDS"), ("OddsDraw", "D_ODDS"), ("OddsAway", "A_ODDS")):
    typed = pd.to_numeric(X[source], errors="coerce")
    X[target] = typed.fillna(X[target])

inverse = 1 / X[["H_ODDS", "D_ODDS", "A_ODDS"]]
X["BOOK"] = inverse["D_ODDS"] / inverse.sum(axis=1)
# The historical model alone chooses the pool draws. Bookmaker prices remain
# a comparison column and must not affect the model's probability or ranking.
X["SCORE"] = X["MODEL"]
X["BASIS"] = np.where(X["BOOK"].notna(), "model; bookmaker is reference only", "model only")
X["LEAGUE"] = X["HomeTeam"].map(team_league)
X["EXPECTED_GOALS"] = (lam_home + lam_away).round(2)

X = X.sort_values("SCORE", ascending=False).reset_index(drop=True)
X["RANK"] = X.index + 1
X["DRAW_PICK"] = X["RANK"] <= DRAW_PICKS

pool_for_scores = X if ALLOW_OVERLAP else X[~X["DRAW_PICK"]]
score_rows = pool_for_scores.sort_values("TOP_SCORE_PROB", ascending=False).head(SCORE_PICKS)
X["SCORE_PICK"] = X.index.isin(score_rows.index)

X["Match"] = X["HomeTeam"] + " vs " + X["AwayTeam"]
X["Draw chance (model)"] = (X["MODEL"] * 100).round(0)
X["Draw chance (bookmaker)"] = (X["BOOK"] * 100).round(0)
X["Draw score"] = (X["SCORE"] * 100).round(1)
X["Likely draw score"] = X["DRAW_SCORE"] + " (" + (X["DRAW_SCORE_PROB"] * 100).round(0).astype(int).astype(str) + "%)"
X["Predicted score"] = X["TOP_SCORE"]
X["Chance of that exact score"] = (X["TOP_SCORE_PROB"] * 100).round(1)

print()
print("Rated", rated, "of", len(pool), "fixtures.", int(X["BOOK"].notna().sum()), "have bookmaker reference prices.")
print()
print("YOUR", DRAW_PICKS, "DRAWS:")
draw_columns = ["RANK", "LEAGUE", "Match", "Draw chance (model)", "Draw chance (bookmaker)",
                "Likely draw score", "EXPECTED_GOALS", "BASIS"]
print(X[X["DRAW_PICK"]][draw_columns].to_string(index=False))
print()
print("YOUR", SCORE_PICKS, "CORRECT SCORES (the fixtures where one exact score is most likely):")
score_columns = ["LEAGUE", "Match", "Predicted score", "Chance of that exact score", "LIKELY_SCORES"]
print(X[X["SCORE_PICK"]][score_columns].to_string(index=False))
print()
print("ALL RATED FIXTURES, from most to least draw-like:")
print(X[draw_columns].to_string(index=False))

X[["RANK", "DRAW_PICK", "SCORE_PICK", "LEAGUE", "HomeTeam", "AwayTeam", "MODEL", "BOOK", "SCORE",
   "DRAW_SCORE", "TOP_SCORE", "TOP_SCORE_PROB", "LIKELY_SCORES", "EXPECTED_GOALS", "BASIS"]].round(3).to_csv(OUT_FILE, index=False)
print()
print("Saved to", OUT_FILE)
