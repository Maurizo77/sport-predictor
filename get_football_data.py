import os
import shutil
import sys
import time

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

LEAGUES = [
    "E0", "E1", "E2", "E3", "EC",
    "SC0", "SC1", "SC2", "SC3",
    "D1", "D2", "I1", "I2", "SP1", "SP2", "F1", "F2",
    "N1", "B1", "P1", "T1", "G1",
]
ALL_SEASONS = ["1920", "2021", "2122", "2223", "2324", "2425", "2526", "2627"]
RECENT = ["2526", "2627"]
KEEP = [
    "SEASON", "LEAGUE", "Date", "HomeTeam", "AwayTeam",
    "FTHG", "FTAG", "FTR", "HTHG", "HTAG", "HTR",
    "HS", "AS", "HST", "AST", "HC", "AC", "HF", "AF", "HO", "AO",
    "HY", "AY", "HR", "AR",
    "B365H", "B365D", "B365A",
]
OUT = "football_games.csv"
BACKUP = "football_games_backup.csv"


def fetch(league, season):
    url = "https://www.football-data.co.uk/mmz4281/" + season + "/" + league + ".csv"
    for attempt in range(1, 4):
        try:
            df = pd.read_csv(url, encoding="latin-1", on_bad_lines="skip")
            df = df.reindex(columns=KEEP).assign(SEASON=season, LEAGUE=league)
            return df.dropna(subset=["FTR"])
        except Exception as error:
            print("  attempt", attempt, "failed:", league, season, type(error).__name__)
            time.sleep(3)
    return None


rebuild_all = len(sys.argv) > 1 and sys.argv[1] == "all"
seasons = ALL_SEASONS if (rebuild_all or not os.path.exists(OUT)) else RECENT
print("Downloading seasons:", seasons)

frames = []
failed = []
for league in LEAGUES:
    for season in seasons:
        df = fetch(league, season)
        if df is None:
            failed.append(league + " " + season)
            print("FAILED", league, season)
        else:
            frames.append(df)
            print("OK", league, season, len(df))
        time.sleep(1)

if not frames:
    print("Nothing could be downloaded. Check your internet connection.")
    raise SystemExit

new = pd.concat(frames)

if os.path.exists(OUT):
    old = pd.read_csv(OUT, dtype={"SEASON": str})
    done = set(new["LEAGUE"] + "|" + new["SEASON"])
    old_keys = old["LEAGUE"] + "|" + old["SEASON"].astype(str)
    data = pd.concat([old[~old_keys.isin(done)], new])
    if len(data) < 0.95 * len(old):
        print("The new file would be much smaller than the old one.")
        print("Not saving. Check your internet connection and run it again.")
        raise SystemExit
    shutil.copyfile(OUT, BACKUP)
else:
    data = new

data.to_csv(OUT, index=False)
print()
print("Saved", len(data), "matches to", OUT)
print("Failed downloads:", failed if failed else "none")
from football_feedback import update_feedback_file

reviewed = update_feedback_file(os.path.dirname(os.path.abspath(__file__)))
print("Completed saved forecasts matched to results:", reviewed)
