import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss

data = pd.read_csv("nba_games.csv")
data["GAME_DATE"] = pd.to_datetime(data["GAME_DATE"])
data = data.sort_values(["TEAM_ID", "GAME_DATE"])

data["WIN"] = (data["WL"] == "W").astype(int)
data["IS_HOME"] = data["MATCHUP"].str.contains("vs.", regex=False)

data["FORM_WIN"] = data.groupby("TEAM_ID")["WIN"].transform(
    lambda s: s.shift(1).rolling(10).mean()
)
data["FORM_MARGIN"] = data.groupby("TEAM_ID")["PLUS_MINUS"].transform(
    lambda s: s.shift(1).rolling(10).mean()
)
data["LONG_MARGIN"] = data.groupby("TEAM_ID")["PLUS_MINUS"].transform(
    lambda s: s.shift(1).rolling(30, min_periods=10).mean()
)
data["REST_DAYS"] = (
    data.groupby("TEAM_ID")["GAME_DATE"].diff().dt.days.clip(upper=7)
)

home = data[data["IS_HOME"]][
    ["GAME_ID", "GAME_DATE", "TEAM_ID", "WIN", "FORM_WIN", "FORM_MARGIN", "LONG_MARGIN", "REST_DAYS"]
]
away = data[~data["IS_HOME"]][
    ["GAME_ID", "TEAM_ID", "FORM_WIN", "FORM_MARGIN", "LONG_MARGIN", "REST_DAYS"]
]

games = home.merge(away, on="GAME_ID", suffixes=("_HOME", "_AWAY"))
games = games.sort_values("GAME_DATE")

ratings = {}
elo_diffs = []

for _, row in games.iterrows():
    home_id = row["TEAM_ID_HOME"]
    away_id = row["TEAM_ID_AWAY"]
    home_rating = ratings.get(home_id, 1500)
    away_rating = ratings.get(away_id, 1500)

    elo_diffs.append((home_rating - away_rating) / 100)

    expected = 1 / (1 + 10 ** ((away_rating - home_rating - 100) / 400))
    change = 20 * (row["WIN"] - expected)
    ratings[home_id] = home_rating + change
    ratings[away_id] = away_rating - change

games["ELO_DIFF"] = elo_diffs
games = games.dropna()

features = [
    "FORM_WIN_HOME", "FORM_MARGIN_HOME", "LONG_MARGIN_HOME", "REST_DAYS_HOME",
    "FORM_WIN_AWAY", "FORM_MARGIN_AWAY", "LONG_MARGIN_AWAY", "REST_DAYS_AWAY",
    "ELO_DIFF",
]

train = games[games["GAME_DATE"] < "2025-10-01"]
test = games[games["GAME_DATE"] >= "2025-10-01"]

model = LogisticRegression(max_iter=1000)
model.fit(train[features], train["WIN"])

predictions = model.predict(test[features])
probs = model.predict_proba(test[features])[:, 1]

print("Always pick home:", round(test["WIN"].mean(), 3))
print("Accuracy:", round(accuracy_score(test["WIN"], predictions), 3))
print("Log loss:", round(log_loss(test["WIN"], probs), 3))

