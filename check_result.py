import pandas as pd
from nba_api.stats.endpoints import leaguegamefinder
from team_names import full

log = pd.read_csv("predictions_log.csv", dtype={"GAME_ID": str})

dates = pd.to_datetime(log["DATE"])
start = dates.min().strftime("%m/%d/%Y")
end = (dates.max() + pd.Timedelta(days=1)).strftime("%m/%d/%Y")

finder = leaguegamefinder.LeagueGameFinder(
    league_id_nullable="00",
    date_from_nullable=start,
    date_to_nullable=end,
)
results = finder.get_data_frames()[0]
results = results[results["GAME_ID"].isin(log["GAME_ID"])]

records = []
correct = 0
finished = 0

print("Prediction results")
for _, row in log.iterrows():
    game = results[results["GAME_ID"] == row["GAME_ID"]]
    is_home = game["MATCHUP"].str.contains("vs.", regex=False)
    home_row = game[is_home]
    away_row = game[~is_home]
    label = full(row["AWAY"]) + " @ " + full(row["HOME"])

    if home_row.empty or away_row.empty or home_row.iloc[0]["WL"] not in ["W", "L"]:
        print(label, "-> not finished yet")
        continue

    winner = row["HOME"] if home_row.iloc[0]["WL"] == "W" else row["AWAY"]
    real_total = home_row.iloc[0]["PTS"] + away_row.iloc[0]["PTS"]
    is_correct = row["PICK"] == winner
    finished += 1
    if is_correct:
        correct += 1
    tick = "✔" if is_correct else "✘"
    print(label, "| picked", full(row["PICK"]), "| winner", full(winner), tick,
          "| total predicted", row["PRED_TOTAL"], "real", real_total)
    records.append({
        "DATE": row["DATE"],
        "GAME_ID": row["GAME_ID"],
        "GAME": label,
        "PICK": full(row["PICK"]),
        "WINNER": full(winner),
        "CORRECT": is_correct,
        "PRED_TOTAL": row["PRED_TOTAL"],
        "REAL_TOTAL": real_total,
    })

print("Score:", correct, "/", finished, "correct")

if records:
    pd.DataFrame(records).to_csv("results_log.csv", index=False)