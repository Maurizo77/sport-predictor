import os
import sys
import tempfile
import time

import numpy as np
import pandas as pd
from scipy.stats import poisson
from sklearn.linear_model import PoissonRegressor
from sklearn.isotonic import IsotonicRegression

from model_cache import load_model_cache, save_model_cache
from football_market_odds import fetch_football_market_odds
from football_feedback import build_feedback
from football_market_features import FEATURE_GROUPS
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
MIN_TOP_PICK_CONFIDENCE = 0.70
CACHE_NAME = "football_training.joblib"
CACHE_VERSION = 4

FORM_TARGETS = {
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

    stats = []
    available_form_targets = {}
    for form_stat, (home_target, away_target) in FORM_TARGETS.items():
        if home_target not in matches.columns or away_target not in matches.columns:
            continue
        home_values = pd.to_numeric(matches[home_target], errors="coerce")
        away_values = pd.to_numeric(matches[away_target], errors="coerce")
        if min(home_values.notna().sum(), away_values.notna().sum()) < 500:
            continue
        stats.append(form_stat)
        available_form_targets[form_stat] = (home_values, away_values)
    if not stats:
        raise ValueError("No historical match statistics are available for team form.")

    home = matches[["MATCH_ID", "Date", "HomeTeam"]].rename(columns={"HomeTeam": "TEAM"}).copy()
    away = matches[["MATCH_ID", "Date", "AwayTeam"]].rename(columns={"AwayTeam": "TEAM"}).copy()
    for form_stat, (home_values, away_values) in available_form_targets.items():
        home[form_stat] = home_values.to_numpy()
        away[form_stat] = away_values.to_numpy()

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

    # Auxiliary match-stat estimates use only pre-match form/Elo features and
    # completed historical targets; bookmaker prices do not enter these models.
    auxiliary_models = {}
    auxiliary_targets = (
        "HC", "AC", "HTHG", "HTAG", "HY", "AY", "HR", "AR", "HS", "AS",
        "HST", "AST", "HF", "AF", "HO", "AO",
    )
    for target in auxiliary_targets:
        if target not in matches.columns:
            continue
        target_values = pd.to_numeric(matches[target], errors="coerce")
        available = (
            target_values.notna() & (target_values >= 0)
            & matches[features].notna().all(axis=1)
        )
        auxiliary_train = matches.loc[available].copy()
        auxiliary_train["_AUX_TARGET"] = target_values.loc[available]
        if len(auxiliary_train) < 500:
            print("Not enough completed data to train the", target, "estimate.")
            continue
        auxiliary_model = PoissonRegressor(alpha=0.01, max_iter=1000)
        auxiliary_model.fit(auxiliary_train[features], auxiliary_train["_AUX_TARGET"])
        auxiliary_models[target] = auxiliary_model

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
        "auxiliary_models": auxiliary_models,
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
auxiliary_models = trained.get("auxiliary_models", {})

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
auxiliary_predictions = {
    target: model.predict(X[features])
    for target, model in auxiliary_models.items()
}


def auxiliary_prediction(target, default=np.nan):
    return auxiliary_predictions.get(target, np.full(len(X), default, dtype=float))


exp_home_corners = auxiliary_prediction("HC")
exp_away_corners = auxiliary_prediction("AC")
exp_home_ht_goals = auxiliary_prediction("HTHG")
exp_away_ht_goals = auxiliary_prediction("HTAG")
exp_home_yellow_cards = auxiliary_prediction("HY")
exp_away_yellow_cards = auxiliary_prediction("AY")
exp_home_red_cards = auxiliary_prediction("HR")
exp_away_red_cards = auxiliary_prediction("AR")
exp_home_shots = auxiliary_prediction("HS")
exp_away_shots = auxiliary_prediction("AS")
exp_home_shots_on_target = auxiliary_prediction("HST")
exp_away_shots_on_target = auxiliary_prediction("AST")
exp_home_fouls = auxiliary_prediction("HF")
exp_away_fouls = auxiliary_prediction("AF")
exp_home_offsides = auxiliary_prediction("HO")
exp_away_offsides = auxiliary_prediction("AO")
has_red_estimates = np.isfinite(exp_home_red_cards) & np.isfinite(exp_away_red_cards)
p_any_red_card = np.full(len(X), np.nan)
red_card_rate = exp_home_red_cards[has_red_estimates] + exp_away_red_cards[has_red_estimates]
p_any_red_card[has_red_estimates] = 1 - np.exp(-red_card_rate)
has_ht_estimates = np.isfinite(exp_home_ht_goals) & np.isfinite(exp_away_ht_goals)
p_first_half_goal = np.full(len(X), np.nan)
p_first_half_2plus = np.full(len(X), np.nan)
first_half_goal_rate = exp_home_ht_goals[has_ht_estimates] + exp_away_ht_goals[has_ht_estimates]
p_first_half_goal[has_ht_estimates] = 1 - np.exp(-first_half_goal_rate)
p_first_half_2plus[has_ht_estimates] = 1 - np.exp(-first_half_goal_rate) * (1 + first_half_goal_rate)
half_goals = np.arange(0, 11)
half_grid = (
    poisson.pmf(half_goals, np.nan_to_num(exp_home_ht_goals)[:, None])[:, :, None]
    * poisson.pmf(half_goals, np.nan_to_num(exp_away_ht_goals)[:, None])[:, None, :]
)
half_i = half_goals[:, None]
half_j = half_goals[None, :]
p_ht_home = (half_grid * (half_i > half_j)).sum(axis=(1, 2))
p_ht_draw = (half_grid * (half_i == half_j)).sum(axis=(1, 2))
p_ht_away = (half_grid * (half_i < half_j)).sum(axis=(1, 2))
p_ht_home[~has_ht_estimates] = np.nan
p_ht_draw[~has_ht_estimates] = np.nan
p_ht_away[~has_ht_estimates] = np.nan

def poisson_over_probability(rate, min_count):
    result = np.full(len(X), np.nan)
    available = np.isfinite(rate) & (rate >= 0)
    result[available] = poisson.sf(min_count - 1, rate[available])
    return result


total_corners = exp_home_corners + exp_away_corners
total_shots = exp_home_shots + exp_away_shots
total_shots_on_target = exp_home_shots_on_target + exp_away_shots_on_target
total_yellow_cards = exp_home_yellow_cards + exp_away_yellow_cards
total_fouls = exp_home_fouls + exp_away_fouls
total_offsides = exp_home_offsides + exp_away_offsides
p_over85_corners = poisson_over_probability(total_corners, 9)
p_over215_shots = poisson_over_probability(total_shots, 22)
p_over75_shots_on_target = poisson_over_probability(total_shots_on_target, 8)
p_over35_yellow_cards = poisson_over_probability(total_yellow_cards, 4)
p_over215_fouls = poisson_over_probability(total_fouls, 22)
p_over25_offsides = poisson_over_probability(total_offsides, 3)

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

# Learn a probability correction from the model's own saved forecasts and
# completed results. Only forecasts from before today are eligible, and the
# raw model probabilities remain in the log so the learning step can be audited.
CALIBRATION_MIN_SAMPLES = 50
historical_feedback = build_feedback(BASE)
today_cutoff = pd.Timestamp.now(tz=FOOTBALL_FEED_TIMEZONE).normalize().tz_localize(None)
if not historical_feedback.empty:
    historical_feedback["DATE"] = pd.to_datetime(historical_feedback["DATE"], errors="coerce")
    historical_feedback = historical_feedback[historical_feedback["DATE"] < today_cutoff]

learned_probabilities = {
    "home_win": p_home.copy(), "draw": p_draw.copy(), "away_win": p_away.copy(),
    "over15": p_o15.copy(), "over25": p_o25_model.copy(), "over35": p_o35.copy(),
    "btts_yes": p_btts_model.copy(),
}
self_calibration_counts = {}
self_calibrated_features = set()
for feature_key in learned_probabilities:
    probability_column = "FEATURE_PROB_" + feature_key
    hit_column = "FEATURE_HIT_" + feature_key
    if probability_column not in historical_feedback or hit_column not in historical_feedback:
        self_calibration_counts[feature_key] = 0
        continue
    history_probability = pd.to_numeric(historical_feedback[probability_column], errors="coerce")
    history_hit = historical_feedback[hit_column].astype("boolean")
    valid = history_probability.notna() & history_hit.notna()
    self_calibration_counts[feature_key] = int(valid.sum())
    if (
        valid.sum() >= CALIBRATION_MIN_SAMPLES
        and history_probability.loc[valid].nunique() >= 5
        and history_hit.loc[valid].nunique() == 2
    ):
        calibrator = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip")
        calibrator.fit(
            history_probability.loc[valid].to_numpy(dtype=float),
            history_hit.loc[valid].astype(int).to_numpy(),
        )
        learned_probabilities[feature_key] = calibrator.predict(
            learned_probabilities[feature_key]
        )
        self_calibrated_features.add(feature_key)

# Keep mutually exclusive 1X2 probabilities coherent after calibration.
learned_1x2_total = sum(learned_probabilities[key] for key in ("home_win", "draw", "away_win"))
for key in ("home_win", "draw", "away_win"):
    learned_probabilities[key] = np.divide(
        learned_probabilities[key], learned_1x2_total,
        out=np.full(len(X), 1 / 3, dtype=float), where=learned_1x2_total > 0,
    )
self_calibrated = bool(self_calibrated_features)
if historical_feedback.empty:
    print("Self-calibration: no earlier completed saved forecasts; raw probabilities will be used.")
elif not self_calibrated:
    largest_sample = max(self_calibration_counts.values(), default=0)
    print(
        "Self-calibration: ", largest_sample,
        "earlier completed forecast(s); at least", CALIBRATION_MIN_SAMPLES,
        "with varied outcomes are required. Raw probabilities will be used.",
        sep="",
    )
else:
    print("Self-calibration applied using earlier completed saved forecasts.")


def pct(x):
    return str(round(x * 100)) + "%"


picks = []

for k in range(len(X)):
    game = X.iloc[k]["HOME"] + " vs " + X.iloc[k]["AWAY"]

    options = {
        "Home win": learned_probabilities["home_win"][k],
        "Draw": learned_probabilities["draw"][k],
        "Away win": learned_probabilities["away_win"][k],
        "Home win or draw (1X)": learned_probabilities["home_win"][k] + learned_probabilities["draw"][k],
        "Draw or away win (X2)": learned_probabilities["draw"][k] + learned_probabilities["away_win"][k],
        "Either team to win (12)": learned_probabilities["home_win"][k] + learned_probabilities["away_win"][k],
        "Over 1.5": learned_probabilities["over15"][k],
        "Under 1.5": 1 - learned_probabilities["over15"][k],
        "Over 2.5": learned_probabilities["over25"][k],
        "Under 2.5": 1 - learned_probabilities["over25"][k],
        "Over 3.5": learned_probabilities["over35"][k],
        "Under 3.5": 1 - learned_probabilities["over35"][k],
        "BTTS yes": learned_probabilities["btts_yes"][k],
        "BTTS no": 1 - learned_probabilities["btts_yes"][k],
    }
    # Keep the legacy one-pick field for overall accuracy history. The app's
    # visible feature sections rank each market independently from the columns below.
    best_name = max(options, key=options.get)
    source = "Poisson model + self-calibration" if self_calibrated else "Poisson model"
    picks.append((options[best_name], game, best_name, k, source))

picks.sort(reverse=True)
print()
print("Fixtures screened:", len(X))
print("Forecasts are grouped by feature in the app; the default feature-list threshold is",
      pct(MIN_TOP_PICK_CONFIDENCE) + ".")

best_by_index = {index: (name, prob, source_name) for prob, _, name, index, source_name in picks}
bet_picks = [best_by_index[k][0] for k in range(len(X))]
bet_probs = [best_by_index[k][1] for k in range(len(X))]
bet_sources = [best_by_index[k][2] for k in range(len(X))]

log = pd.DataFrame({
    "DATE": X["DATE"], "EVENT_TIME_WAT": X["EVENT_TIME_WAT"],
    "LEAGUE": X["LEAGUE"], "HOME": X["HOME"], "AWAY": X["AWAY"],
    "P_HOME": p_home.round(3), "P_DRAW": p_draw.round(3), "P_AWAY": p_away.round(3),
    "P_HOME_OR_DRAW": (p_home + p_draw).round(3),
    "P_DRAW_OR_AWAY": (p_draw + p_away).round(3),
    "P_HOME_OR_AWAY": (p_home + p_away).round(3),
    "P_HOME_MODEL": p_home_model.round(3), "P_DRAW_MODEL": p_draw_model.round(3),
    "P_AWAY_MODEL": p_away_model.round(3),
    "P_HOME_MARKET": market_probs[:, 0].round(3),
    "P_DRAW_MARKET": market_probs[:, 1].round(3),
    "P_AWAY_MARKET": market_probs[:, 2].round(3),
    "P_OVER15": p_o15.round(3), "P_UNDER15": (1 - p_o15).round(3),
    "P_OVER25": p_o25_model.round(3), "P_UNDER25": (1 - p_o25_model).round(3),
    "P_OVER35": p_o35.round(3), "P_UNDER35": (1 - p_o35).round(3),
    "P_BTTS": p_btts_model.round(3), "P_BTTS_NO": (1 - p_btts_model).round(3),
    "P_OVER25_MARKET": p_o25_market.round(3), "P_BTTS_MARKET": p_btts_market.round(3),
    "OVER25_BOOKS": pd.to_numeric(X["OVER25_BOOKS"], errors="coerce"),
    "BTTS_BOOKS": pd.to_numeric(X["BTTS_BOOKS"], errors="coerce"),
    "OVER25_SOURCE": X["OVER25_SOURCE"], "BTTS_SOURCE": X["BTTS_SOURCE"],
    "EXP_HOME_GOALS": lam_home.round(2), "EXP_AWAY_GOALS": lam_away.round(2),
    "EXP_HOME_CORNERS": np.round(exp_home_corners, 2),
    "EXP_AWAY_CORNERS": np.round(exp_away_corners, 2),
    "P_OVER85_CORNERS": np.round(p_over85_corners, 3),
    "P_UNDER85_CORNERS": np.round(1 - p_over85_corners, 3),
    "EXP_HOME_SHOTS": np.round(exp_home_shots, 2),
    "EXP_AWAY_SHOTS": np.round(exp_away_shots, 2),
    "P_OVER215_SHOTS": np.round(p_over215_shots, 3),
    "P_UNDER215_SHOTS": np.round(1 - p_over215_shots, 3),
    "EXP_HOME_SHOTS_ON_TARGET": np.round(exp_home_shots_on_target, 2),
    "EXP_AWAY_SHOTS_ON_TARGET": np.round(exp_away_shots_on_target, 2),
    "EXP_HOME_FOULS": np.round(exp_home_fouls, 2), "EXP_AWAY_FOULS": np.round(exp_away_fouls, 2),
    "EXP_HOME_OFFSIDES": np.round(exp_home_offsides, 2), "EXP_AWAY_OFFSIDES": np.round(exp_away_offsides, 2),
    "P_OVER75_SHOTS_ON_TARGET": np.round(p_over75_shots_on_target, 3),
    "P_UNDER75_SHOTS_ON_TARGET": np.round(1 - p_over75_shots_on_target, 3),
    "P_OVER215_FOULS": np.round(p_over215_fouls, 3),
    "P_UNDER215_FOULS": np.round(1 - p_over215_fouls, 3),
    "P_OVER25_OFFSIDES": np.round(p_over25_offsides, 3),
    "P_UNDER25_OFFSIDES": np.round(1 - p_over25_offsides, 3),
    "EXP_HOME_HT_GOALS": np.round(exp_home_ht_goals, 2),
    "EXP_AWAY_HT_GOALS": np.round(exp_away_ht_goals, 2),
    "P_FIRST_HALF_GOAL": np.round(p_first_half_goal, 3),
    "P_NO_FIRST_HALF_GOAL": np.round(1 - p_first_half_goal, 3),
    "P_FIRST_HALF_2PLUS": np.round(p_first_half_2plus, 3),
    "P_FIRST_HALF_UNDER15": np.round(1 - p_first_half_2plus, 3),
    "P_HT_HOME": np.round(p_ht_home, 3), "P_HT_DRAW": np.round(p_ht_draw, 3),
    "P_HT_AWAY": np.round(p_ht_away, 3),
    "EXP_HOME_YELLOW_CARDS": np.round(exp_home_yellow_cards, 2),
    "EXP_AWAY_YELLOW_CARDS": np.round(exp_away_yellow_cards, 2),
    "P_OVER35_YELLOW_CARDS": np.round(p_over35_yellow_cards, 3),
    "P_UNDER35_YELLOW_CARDS": np.round(1 - p_over35_yellow_cards, 3),
    "EXP_HOME_RED_CARDS": np.round(exp_home_red_cards, 3),
    "EXP_AWAY_RED_CARDS": np.round(exp_away_red_cards, 3),
    "P_ANY_RED_CARD": np.round(p_any_red_card, 3),
    "P_NO_RED_CARD": np.round(1 - p_any_red_card, 3),
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
print("Selections meeting the default feature confidence threshold:")
saved_forecasts = pd.read_csv(LOG_FILE)
saved_forecasts["DATE"] = pd.to_datetime(saved_forecasts["DATE"], errors="coerce")
saved_forecasts = saved_forecasts[saved_forecasts["DATE"] >= today]
for group, markets in FEATURE_GROUPS.items():
    count = 0
    match_keys = set()
    for _key, _label, probability_column, _actual_event in markets:
        if probability_column not in saved_forecasts.columns:
            continue
        probability = pd.to_numeric(saved_forecasts[probability_column], errors="coerce")
        selected = saved_forecasts.loc[probability >= MIN_TOP_PICK_CONFIDENCE]
        count += len(selected)
        match_keys.update(
            selected["DATE"].astype(str) + "|" + selected["LEAGUE"].astype(str) + "|"
            + selected["HOME"].astype(str) + "|" + selected["AWAY"].astype(str)
        )
    print(" ", group + ":", count, "selections across", len(match_keys), "matches")
