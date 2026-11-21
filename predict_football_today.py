import os
import sys
import tempfile
import time

import numpy as np
import pandas as pd
from scipy.stats import poisson
from sklearn.linear_model import PoissonRegressor

from model_cache import load_model_cache, save_model_cache
from football_market_odds import fetch_football_market_odds
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
FIXTURE_CACHE = os.path.join(BASE, ".cache", "football_fixtures.csv")
MAX_FIXTURE_CACHE_AGE_DAYS = 3
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

manual_path = os.path.join(BASE, MANUAL_FILE)
if os.path.exists(manual_path):
    print("Fixtures from manual file:", manual_path)
    try:
        fixtures = pd.read_csv(
            manual_path, encoding="utf-8-sig", encoding_errors="replace", on_bad_lines="skip"
        )
    except Exception as error:
        print("Could not read the manual fixtures:", type(error).__name__, str(error))
        raise SystemExit
else:
    print("Fixtures from:", FIXTURES_URL)
    try:
        fixtures = pd.read_csv(
            FIXTURES_URL, encoding="utf-8-sig", encoding_errors="replace", on_bad_lines="skip"
        )
        try:
            os.makedirs(os.path.dirname(FIXTURE_CACHE), exist_ok=True)
            fixtures.to_csv(FIXTURE_CACHE, index=False)
        except OSError as cache_error:
            print("The live fixtures loaded, but the local cache could not be saved:", type(cache_error).__name__)
    except Exception as error:
        print("Could not download the fixture feed:", type(error).__name__, str(error))
        cache_available = os.path.exists(FIXTURE_CACHE)
        cache_age = None
        if cache_available:
            cache_age = pd.Timestamp.now() - pd.Timestamp.fromtimestamp(
                os.path.getmtime(FIXTURE_CACHE)
            )
        if cache_available and cache_age <= pd.Timedelta(days=MAX_FIXTURE_CACHE_AGE_DAYS):
            fixtures = pd.read_csv(FIXTURE_CACHE, encoding="utf-8-sig", on_bad_lines="skip")
            print(
                "Using saved fixtures updated",
                round(cache_age.total_seconds() / 3600, 1),
                "hours ago. Please check for postponed or changed games.",
            )
        else:
            print("Check that this computer has internet access, then run the command again.")
            if cache_available:
                print("The saved fixture cache is older than 3 days, so it was not used.")
            print("Offline option: place a current my_fixtures.csv in the project folder.")
            print("It must contain Div, Date, HomeTeam and AwayTeam columns; Time is optional.")
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
        "ODDS_HOME": getattr(fx, "B365H", getattr(fx, "OddsHome", np.nan)),
        "ODDS_DRAW": getattr(fx, "B365D", getattr(fx, "OddsDraw", np.nan)),
        "ODDS_AWAY": getattr(fx, "B365A", getattr(fx, "OddsAway", np.nan)),
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

market_odds = fetch_football_market_odds(X)
X = pd.concat([X, market_odds], axis=1)
if market_odds["P_OVER25_MARKET"].notna().any():
    print("Bookmaker over/under 2.5 odds found for", int(market_odds["P_OVER25_MARKET"].notna().sum()), "matches.")
if market_odds["P_BTTS_MARKET"].notna().any():
    print("Bookmaker BTTS odds found for", int(market_odds["P_BTTS_MARKET"].notna().sum()), "matches.")

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
# Keep the model's estimates as the sole basis for predictions. Odds below are
# stored as comparison references and never replace or rank the model's picks.
p_home_model, p_draw_model, p_away_model = p_home.copy(), p_draw.copy(), p_away.copy()

# Keep no-margin bookmaker probabilities as an independent market reference.
odds = X[["ODDS_HOME", "ODDS_DRAW", "ODDS_AWAY"]].apply(
    pd.to_numeric, errors="coerce"
).to_numpy(dtype=float)
odds_available = np.isfinite(odds).all(axis=1) & (odds > 1).all(axis=1)
market_probs = np.full_like(odds, np.nan, dtype=float)
if odds_available.any():
    implied = 1 / odds[odds_available]
    market_probs[odds_available] = implied / implied.sum(axis=1, keepdims=True)
outcome_source = np.full(len(X), "Poisson model", dtype=object)
p_o15 = (grid * (i + j > 1.5)).sum(axis=(1, 2))
p_o25 = (grid * (i + j > 2.5)).sum(axis=(1, 2))
p_o35 = (grid * (i + j > 3.5)).sum(axis=(1, 2))
p_btts = (grid * ((i > 0) & (j > 0))).sum(axis=(1, 2))
p_o25_model, p_btts_model = p_o25.copy(), p_btts.copy()
p_o25_market = pd.to_numeric(X["P_OVER25_MARKET"], errors="coerce").to_numpy(dtype=float)
p_btts_market = pd.to_numeric(X["P_BTTS_MARKET"], errors="coerce").to_numpy(dtype=float)
has_o25_market = np.isfinite(p_o25_market)
has_btts_market = np.isfinite(p_btts_market)


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
    print("  Model 1X2 estimates (not certainties): " + pct(p_home[k]) + " / " + pct(p_draw[k]) + " / " + pct(p_away[k]))
    print("  Double chance 1X / X2 / 12: " + pct(p_home[k] + p_draw[k]) + " / "
          + pct(p_draw[k] + p_away[k]) + " / " + pct(p_home[k] + p_away[k]))
    print("  Over 1.5 / 2.5 / 3.5 goals: " + pct(p_o15[k]) + " / " + pct(p_o25[k]) + " / " + pct(p_o35[k]))
    print("  Both teams score: " + pct(p_btts[k]))
    if has_o25_market[k]:
        print("  Bookmaker Over 2.5 estimate:", pct(p_o25_market[k]), "|", X.iloc[k]["OVER25_SOURCE"])
    if has_btts_market[k]:
        print("  Bookmaker BTTS estimate:", pct(p_btts_market[k]), "|", X.iloc[k]["BTTS_SOURCE"])
    if odds_available[k]:
        print("  Bookmaker 1X2 reference:", pct(market_probs[k, 0]), "/", pct(market_probs[k, 1]), "/", pct(market_probs[k, 2]))
    print("  Likely scores: " + top_scores)

    options = {
        "Home win": p_home[k], "Draw": p_draw[k], "Away win": p_away[k],
        "Home or draw": p_home[k] + p_draw[k],
        "Draw or away": p_draw[k] + p_away[k],
        "Home or away": p_home[k] + p_away[k],
        "Over 1.5": p_o15[k], "Under 1.5": 1 - p_o15[k],
        "Over 2.5": p_o25_model[k], "Under 2.5": 1 - p_o25_model[k],
        "Over 3.5": p_o35[k], "Under 3.5": 1 - p_o35[k],
        "BTTS yes": p_btts_model[k], "BTTS no": 1 - p_btts_model[k],
    }
    # Keep one highest-probability option per match so the shortlist cannot
    # be filled with several overlapping markets from the same fixture.
    best_name = max(options, key=options.get)
    picks.append((options[best_name], game, best_name, k, "Poisson model"))

picks.sort(reverse=True)
print()
print("Highest estimated chances (probabilities are uncertain, not guarantees):")
shortlist = picks[:12]
for prob, game, name, _, source_name in shortlist:
    print(" ", pct(prob), game, "->", name, "| source", source_name, "| model fair-odds estimate", round(1 / prob, 2))

best_by_index = {index: (name, prob, source_name) for prob, _, name, index, source_name in picks}
bet_picks = [best_by_index[k][0] for k in range(len(X))]
bet_probs = [best_by_index[k][1] for k in range(len(X))]
bet_sources = [best_by_index[k][2] for k in range(len(X))]

log = pd.DataFrame({
    "DATE": X["DATE"], "EVENT_TIME_WAT": X["EVENT_TIME_WAT"],
    "LEAGUE": X["LEAGUE"], "HOME": X["HOME"], "AWAY": X["AWAY"],
    "P_HOME": p_home.round(3), "P_DRAW": p_draw.round(3), "P_AWAY": p_away.round(3),
    "P_HOME_MODEL": p_home_model.round(3), "P_DRAW_MODEL": p_draw_model.round(3),
    "P_AWAY_MODEL": p_away_model.round(3),
    "P_HOME_MARKET": market_probs[:, 0].round(3),
    "P_DRAW_MARKET": market_probs[:, 1].round(3),
    "P_AWAY_MARKET": market_probs[:, 2].round(3),
    "P_OVER25": p_o25_model.round(3), "P_BTTS": p_btts_model.round(3),
    "P_OVER25_MARKET": p_o25_market.round(3), "P_BTTS_MARKET": p_btts_market.round(3),
    "OVER25_BOOKS": pd.to_numeric(X["OVER25_BOOKS"], errors="coerce"),
    "BTTS_BOOKS": pd.to_numeric(X["BTTS_BOOKS"], errors="coerce"),
    "OVER25_SOURCE": X["OVER25_SOURCE"], "BTTS_SOURCE": X["BTTS_SOURCE"],
    "EXP_HOME_GOALS": lam_home.round(2), "EXP_AWAY_GOALS": lam_away.round(2),
    "ODDS_SOURCE": outcome_source,
    "MARKET_1X2_SOURCE": np.where(odds_available, "Fixture bookmaker odds reference", ""),
    "BEST_PICK": bet_picks, "BEST_PICK_PROB": np.round(bet_probs, 3),
    "BEST_PICK_SOURCE": bet_sources,
})
if os.path.exists(LOG_FILE):
    old = pd.read_csv(LOG_FILE)
    old_keys = (
        old["DATE"].astype(str) + "|" + old["LEAGUE"].astype(str) + "|"
        + old["HOME"].astype(str) + "|" + old["AWAY"].astype(str)
    )
    new_keys = (
        log["DATE"].astype(str) + "|" + log["LEAGUE"].astype(str) + "|"
        + log["HOME"].astype(str) + "|" + log["AWAY"].astype(str)
    )
    local_today = pd.Timestamp.now(tz=FOOTBALL_FEED_TIMEZONE).date()
    old_dates = pd.to_datetime(old["DATE"], errors="coerce").dt.date
    locked_keys = set(old_keys[old_dates <= local_today])
    # Keep the first saved forecast for today and earlier so result updates
    # cannot rewrite a finished match's original prediction.
    log = log[~new_keys.isin(locked_keys)]
    new_keys = new_keys.loc[log.index]
    old = old[~(old_keys.isin(new_keys) & (old_dates > local_today))]
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
from football_feedback import update_feedback_file

reviewed = update_feedback_file(BASE)
print("Completed saved forecasts matched to results:", reviewed)
print()
print("Saved or refreshed", len(log), "predictions to", LOG_FILE)
