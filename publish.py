import base64
import hashlib
import os
import sys

import requests

sys.stdout.reconfigure(encoding="utf-8")

OWNER = "Maurizo77"
REPO = "sport-predictor"
BRANCH = "main"
FILES = [
    "app.py",
    "app_online.py",
    "football_predictions_view.py",
    "football_market_odds.py",
    "football_feedback.py",
    "football_market_features.py",
    "picks_view.py",
    "results_view.py",
    "team_names.py",
    "predict_today.py",
    "predict_football_today.py",
    "get_football_data.py",
    "fetch_pool_fixtures.py",
    "refresh_pool.py",
    "make_pool_file.py",
    "pool_draws.py",
    "event_time.py",
    "model_cache.py",
    "publish.py",
    "README.md",
    ".gitignore",
    "start_app.bat.txt.bat",
    "research/README.md",
    "research/backtests/compare_football_models.py",
    "requirements.txt",
    "predictions_log.csv",
    "results_log.csv",
    "injuries_log.csv",
    "football_predictions_log.csv",
    "football_feedback_log.csv",
    "pool_picks.csv",
]

requested_files = sys.argv[1:]
unknown_files = sorted(set(requested_files) - set(FILES))
if unknown_files:
    print("Cannot publish unlisted files:", ", ".join(unknown_files))
    raise SystemExit(2)
files_to_publish = requested_files or FILES

token = os.environ.get("GITHUB_TOKEN")
if not token:
    print("No GITHUB_TOKEN found. Set it with setx, then reopen VS Code.")
    raise SystemExit

headers = {
    "Authorization": "Bearer " + token,
    "Accept": "application/vnd.github+json",
    "User-Agent": "sports-predictor-publisher",
}

for name in files_to_publish:
    if not os.path.exists(name):
        print("Skipped (file not found):", name)
        continue
    if os.path.getsize(name) == 0:
        print("Skipped (file is empty):", name)
        continue

    with open(name, "rb") as f:
        raw = f.read()
    local_sha = hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()

    url = "https://api.github.com/repos/" + OWNER + "/" + REPO + "/contents/" + name
    existing = requests.get(url, headers=headers, params={"ref": BRANCH}, timeout=30)

    body = {
        "message": "Update " + name,
        "content": base64.b64encode(raw).decode("ascii"),
        "branch": BRANCH,
    }
    if existing.status_code == 200:
        if existing.json()["sha"] == local_sha:
            print("Unchanged:", name)
            continue
        body["sha"] = existing.json()["sha"]
    elif existing.status_code != 404:
        print("Could not check", name, "- status", existing.status_code)
        continue

    result = requests.put(url, headers=headers, json=body, timeout=60)
    if result.status_code in (200, 201):
        print("Uploaded:", name)
    else:
        print("Failed:", name, "- status", result.status_code, result.json().get("message"))
