import sys
from datetime import datetime, timezone

import pandas as pd
import requests

sys.stdout.reconfigure(encoding="utf-8")
pd.set_option("display.width", 200)

URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/injuries"

response = requests.get(URL, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
response.raise_for_status()
data = response.json()

rows = []
for team in data.get("injuries", []):
    for item in team.get("injuries", []):
        athlete = item.get("athlete", {})
        details = item.get("details", {})
        rows.append({
            "TEAM": team.get("displayName"),
            "PLAYER": athlete.get("displayName"),
            "POSITION": athlete.get("position", {}).get("abbreviation"),
            "STATUS": item.get("status"),
            "PROBLEM": details.get("type"),
            "RETURN": details.get("returnDate"),
            "NOTE": item.get("shortComment"),
            "UPDATED": item.get("date"),
        })

table = pd.DataFrame(rows)
table["FETCHED"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
table.to_csv("injuries_log.csv", index=False)

print("Players on the report:", len(table))
print(table["STATUS"].value_counts())
print()
print("Teams with the most players listed as Out:")
out_counts = table[table["STATUS"] == "Out"].groupby("TEAM").size()
print(out_counts.sort_values(ascending=False).head(8))
print()
print(table[["TEAM", "PLAYER", "STATUS", "RETURN"]].head(12).to_string(index=False))

