import pandas as pd

data = pd.read_csv("nba_games.csv")
data["GAME_DATE"] = pd.to_datetime(data["GAME_DATE"])
data = data.sort_values(["TEAM_ID", "GAME_DATE"])

data["WIN"] = (data["WL"] == "W").astype(int)

data["FORM_WIN"] = data.groupby("TEAM_ID")["WIN"].transform(
    lambda s: s.shift(1).rolling(10).mean()
)
data["FORM_MARGIN"] = data.groupby("TEAM_ID")["PLUS_MINUS"].transform(
    lambda s: s.shift(1).rolling(10).mean()
)
data["REST_DAYS"] = data.groupby("TEAM_ID")["GAME_DATE"].diff().dt.days

print(data[["GAME_DATE", "TEAM_NAME", "WIN", "FORM_WIN", "FORM_MARGIN", "REST_DAYS"]].head(15))