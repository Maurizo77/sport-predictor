import os
import time

import pandas as pd
from nba_api.live.nba.endpoints import scoreboard
from sklearn.linear_model import LinearRegression, LogisticRegression

from model_cache import load_model_cache, save_model_cache
from team_names import full


BASE = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(BASE, "nba_games.csv")
LOG_FILE = os.path.join(BASE, "predictions_log.csv")
CACHE_NAME = "nba_models.joblib"
CACHE_VERSION = 1


def train_models():
    data = pd.read_csv(DATA_FILE)
    data["GAME_DATE"] = pd.to_datetime(data["GAME_DATE"])
    data = data.sort_values(["TEAM_ID", "GAME_DATE"])

    data["IS_HOME"] = data["MATCHUP"].str.contains("vs.", regex=False)
    data["WIN"] = (data["WL"] == "W").astype(int)
    data["PTS_AGAINST"] = data["PTS"] - data["PLUS_MINUS"]
    data["POSS"] = data["FGA"] + 0.44 * data["FTA"] - data["OREB"] + data["TOV"]
    data["REST_DAYS"] = (
        data.groupby("TEAM_ID")["GAME_DATE"].diff().dt.days.clip(upper=7)
    )

    def rolling(column, window, min_periods, shift):
        return data.groupby("TEAM_ID")[column].transform(
            lambda series: series.shift(shift).rolling(window, min_periods=min_periods).mean()
        )

    def add_features(shift, tag):
        data["FORM_WIN" + tag] = rolling("WIN", 10, 10, shift)
        data["FORM_MARGIN" + tag] = rolling("PLUS_MINUS", 10, 10, shift)
        data["LONG_MARGIN" + tag] = rolling("PLUS_MINUS", 30, 10, shift)
        data["AVG_PTS" + tag] = rolling("PTS", 10, 5, shift)
        data["AVG_PTS_AGAINST" + tag] = rolling("PTS_AGAINST", 10, 5, shift)
        data["AVG_POSS" + tag] = rolling("POSS", 10, 5, shift)

    add_features(1, "")
    add_features(0, "_NOW")

    stats = [
        "FORM_WIN", "FORM_MARGIN", "LONG_MARGIN", "AVG_PTS",
        "AVG_PTS_AGAINST", "AVG_POSS",
    ]
    cols = stats + ["REST_DAYS"]
    home_rows = data[data["IS_HOME"]][
        ["GAME_ID", "GAME_DATE", "TEAM_ID", "WIN", "PTS"] + cols
    ]
    away_rows = data[~data["IS_HOME"]][["GAME_ID", "TEAM_ID", "PTS"] + cols]
    games = home_rows.merge(away_rows, on="GAME_ID", suffixes=("_HOME", "_AWAY"))
    games = games.sort_values("GAME_DATE")

    ratings = {}
    elo_diff = []
    for game in games.itertuples():
        rating_home = ratings.get(game.TEAM_ID_HOME, 1500)
        rating_away = ratings.get(game.TEAM_ID_AWAY, 1500)
        elo_diff.append((rating_home - rating_away) / 100)
        expected = 1 / (1 + 10 ** ((rating_away - rating_home - 100) / 400))
        change = 20 * (game.WIN - expected)
        ratings[game.TEAM_ID_HOME] = rating_home + change
        ratings[game.TEAM_ID_AWAY] = rating_away - change
    games["ELO_DIFF"] = elo_diff

    games = games.dropna()
    games["TOTAL"] = games["PTS_HOME"] + games["PTS_AWAY"]
    win_names = ["FORM_WIN", "FORM_MARGIN", "LONG_MARGIN", "REST_DAYS"]
    total_names = ["AVG_PTS", "AVG_PTS_AGAINST", "AVG_POSS", "REST_DAYS"]
    win_features = [n + s for n in win_names for s in ["_HOME", "_AWAY"]] + ["ELO_DIFF"]
    total_features = [n + s for n in total_names for s in ["_HOME", "_AWAY"]]

    win_model = LogisticRegression(max_iter=1000).fit(games[win_features], games["WIN"])
    total_model = LinearRegression().fit(games[total_features], games["TOTAL"])
    latest = data.groupby("TEAM_ID").tail(1).set_index("TEAM_ABBREVIATION")

    return {
        "data_up_to": data["GAME_DATE"].max().date().isoformat(),
        "stats": stats,
        "win_features": win_features,
        "total_features": total_features,
        "ratings": ratings,
        "latest": latest,
        "win_model": win_model,
        "total_model": total_model,
    }


pipeline_started = time.perf_counter()
trained = load_model_cache(CACHE_NAME, DATA_FILE, CACHE_VERSION)
if trained is None:
    print("Training NBA models from game history...")
    trained = train_models()
    save_model_cache(CACHE_NAME, DATA_FILE, CACHE_VERSION, trained)
else:
    print("Loaded cached NBA models; game history has not changed.")
print("Model pipeline time:", round(time.perf_counter() - pipeline_started, 2), "seconds")

stats = trained["stats"]
win_features = trained["win_features"]
total_features = trained["total_features"]
ratings = trained["ratings"]
latest = trained["latest"]
win_model = trained["win_model"]
total_model = trained["total_model"]

board = scoreboard.ScoreBoard().get_dict()["scoreboard"]
today = pd.to_datetime(board["gameDate"])
rows = []
for game in board["games"]:
    if game["gameStatus"] != 1:
        continue
    home = game["homeTeam"]["teamTricode"]
    away = game["awayTeam"]["teamTricode"]
    if home not in latest.index or away not in latest.index:
        print("Skipping", away, "@", home, "(team not found in data)")
        continue
    h = latest.loc[home]
    a = latest.loc[away]
    row = {"GAME_ID": game["gameId"], "HOME": home, "AWAY": away}
    for column in stats:
        row[column + "_HOME"] = h[column + "_NOW"]
        row[column + "_AWAY"] = a[column + "_NOW"]
    row["REST_DAYS_HOME"] = min((today - h["GAME_DATE"]).days, 7)
    row["REST_DAYS_AWAY"] = min((today - a["GAME_DATE"]).days, 7)
    row["ELO_DIFF"] = (
        ratings.get(h["TEAM_ID"], 1500) - ratings.get(a["TEAM_ID"], 1500)
    ) / 100
    rows.append(row)

if not rows:
    print("No upcoming games to predict right now.")
    raise SystemExit

predictions = pd.DataFrame(rows)
win_prob = win_model.predict_proba(predictions[win_features])[:, 1]
total_pred = total_model.predict(predictions[total_features])
predictions["PICK"] = [
    home if probability >= 0.5 else away
    for home, away, probability in zip(
        predictions["HOME"], predictions["AWAY"], win_prob
    )
]
predictions["HOME_WIN_PROB"] = win_prob.round(3)
predictions["PRED_TOTAL"] = total_pred.round(1)

print("Data up to:", trained["data_up_to"])
print("Predictions for", board["gameDate"])
for _, row in predictions.iterrows():
    confidence = max(row["HOME_WIN_PROB"], 1 - row["HOME_WIN_PROB"])
    print(
        full(row["AWAY"]), "@", full(row["HOME"]), "->", full(row["PICK"]),
        str(round(confidence * 100)) + "%", "| total", row["PRED_TOTAL"],
    )

log = predictions[
    ["GAME_ID", "HOME", "AWAY", "PICK", "HOME_WIN_PROB", "PRED_TOTAL"]
].copy()
log.insert(0, "DATE", board["gameDate"])
if os.path.exists(LOG_FILE):
    old = pd.read_csv(LOG_FILE, dtype={"GAME_ID": str})
    log = log[~log["GAME_ID"].isin(old["GAME_ID"])]
log.to_csv(LOG_FILE, mode="a", header=not os.path.exists(LOG_FILE), index=False)
print("Saved", len(log), "new predictions to", LOG_FILE)
