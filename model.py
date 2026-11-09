import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

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
data["REST_DAYS"] = (
    data.groupby("TEAM_ID")["GAME_DATE"].diff().dt.days.clip(upper=7)
)

home = data[data["IS_HOME"]][
    ["GAME_ID", "GAME_DATE", "WIN", "FORM_WIN", "FORM_MARGIN", "REST_DAYS"]
]
away = data[~data["IS_HOME"]][
    ["GAME_ID", "FORM_WIN", "FORM_MARGIN", "REST_DAYS"]
]

games = home.merge(away, on="GAME_ID", suffixes=("_HOME", "_AWAY"))
games = games.dropna().sort_values("GAME_DATE")

features = [
    "FORM_WIN_HOME", "FORM_MARGIN_HOME", "REST_DAYS_HOME",
    "FORM_WIN_AWAY", "FORM_MARGIN_AWAY", "REST_DAYS_AWAY",
]

train = games[games["GAME_DATE"] < "2023-10-01"]
test = games[games["GAME_DATE"] >= "2023-10-01"]

model = LogisticRegression(max_iter=1000)
model.fit(train[features], train["WIN"])

predictions = model.predict(test[features])

print("Games to test on:", len(test))
print("Always pick home:", round(test["WIN"].mean(), 3))
print("Our model:", round(accuracy_score(test["WIN"], predictions), 3))

from sklearn.metrics import log_loss

probs = model.predict_proba(test[features])[:, 1]
print("Log loss:", round(log_loss(test["WIN"], probs), 3))
print(probs[:5].round(2))


seasons = ["2019-20", "2020-21", "2021-22", "2022-23", "2023-24", "2024-25", "2025-26", "2026-27"]