# Sports Predictor

A local Streamlit dashboard for NBA and football predictions. NBA picks and results are saved in CSV files. Football predictions use recent team form and Elo ratings to estimate match and goal probabilities. These are statistical estimates, not guarantees.

## Start on Windows

From this folder, create and activate a Python environment, then install the dependencies:

```powershell
py -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Start the dashboard with:

```powershell
python -m streamlit run app.py
```

Or double-click `start_app.bat.txt.bat`. It expects the environment above in the `venv` folder and opens the dashboard in Microsoft Edge.

## Dashboard actions

- **Basketball (NBA):** make today's picks, review saved results, refresh the NBA game history, and refresh injury data.
- **Football:** predict the broader set of fixtures available in the football-data.co.uk feed, independently of the 49-match pool coupon. The model ranks its own market estimates; bookmaker prices are shown only as references.
- **Pool draws:** use **Fetch next coupon and refresh draw picks** to load the next upcoming 49-match coupon from AbleFAST automatically. The manual fallback reads one fixture per line from `pool_list.txt`. This separate workflow selects exactly two draw candidates from that coupon using the model's draw probabilities; bookmaker draw prices are references only.
- **Publish phone app files:** uploads the phone app, its shared view code, dependencies, and current prediction/result logs to the GitHub repository configured in `publish.py`. Set `GITHUB_TOKEN` before using it. This makes remote changes. Your hosting provider must be connected to that repository for the site to redeploy; the upload itself does not create or host a website.

The first data refresh and predictions require an internet connection. NBA predictions read `nba_games.csv`; football predictions read `football_games.csv`. The football predictor caches the fixture feed locally and can use a cache up to three days old during a connection outage; it warns that changed or postponed matches should be checked. A current `my_fixtures.csv` can be used instead, with `Div`, `Date`, `HomeTeam`, and `AwayTeam` columns and optional `Time`. The update scripts save refreshed data locally, and the football updater keeps `football_games_backup.csv` before replacing data. For each new weekly pool coupon, replace the prior fixtures in `pool_list.txt` before refreshing the pool picks.

The NBA predictor stores its trained models and prepared team state in `.cache/`. The football predictor stores its trained models, Elo ratings, and latest team form there. The cache is local, ignored by Git, and refreshed automatically when the corresponding history CSV changes. Football forecasts are compared with completed results from `football_games.csv`; refreshing that history also updates the feedback log. The model learns from actual match results on the next prediction run, not from its own guesses. Prediction output includes the time spent preparing or loading the cached training pipeline.

## Match times

Both prediction screens show event times in Nigerian time (WAT, Africa/Lagos). NBA start times are converted from the scoreboard's UTC timestamp. Football fixture-feed times are treated as UK local time and converted with daylight-saving rules; a manual `my_fixtures.csv` can include an optional `TimeZone` column with an IANA zone name if its `Time` values use another local timezone. Football predictions are generated from the historical model; 1X2, totals and BTTS bookmaker probabilities are displayed separately as references and do not choose or alter the model's picks. Missing or unparseable times display as `Time unavailable`. Older saved predictions remain visible and show that label until new predictions are generated.

## Data and generated files

- `nba_games.csv`, `football_games.csv`: model history used for predictions.
- `predictions_log.csv`, `football_predictions_log.csv`: saved predictions.
- `football_feedback_log.csv`: saved football predictions matched to completed historical results for the accuracy report.
- `results_log.csv`, `injuries_log.csv`: NBA result checks and latest injury report.
- `pool_list.txt`, `pool_fixtures.csv`, `pool_picks.csv`: optional football pool workflow (`make_pool_file.py`, then `pool_draws.py`).
- `*_backup.csv`: local backups; keep these if you need to restore prior data.

The CSV history and logs are local project data. They are intentionally not excluded by `.gitignore`; review them before committing or sharing the folder.

## Main scripts

| File | Purpose |
| --- | --- |
| `app.py` | Main local Streamlit dashboard |
| `app_online.py` | Read-only dashboard intended for phone-sized screens |
| `football_predictions_view.py` | Shared, compact football predictions table used by both dashboards |
| `predict_today.py`, `predict_football_today.py` | Train and save current NBA and football predictions |
| `model_cache.py` | Reuse trained models while the source game history stays unchanged |
| `event_time.py` | Convert NBA and football kickoff times to Nigerian time |
| `check_result.py` | Fetch NBA outcomes and update the results log |
| `get_all_seasons.py`, `get_football_data.py` | Refresh NBA and football history |
| `injuries.py` | Refresh the NBA injury report |
| `publish.py` | Upload selected dashboard and log files to GitHub |
| `make_pool_file.py`, `pool_draws.py` | Build and rate a football pool coupon |

Backtests and data-inspection scripts are grouped under `research/`; they are not needed to start the dashboard. Run them from the project root so their relative CSV paths resolve correctly. The supported pool builder is `make_pool_file.py`; a duplicate legacy copy is kept under `research/legacy/` for reference.

## Optional odds lookup

`fetch_lines.py` is separate from the dashboard and needs an Odds API key in `ODDS_API_KEY`. Football predictions also use that key to request bookmaker totals and, when offered, BTTS prices from The Odds API. The app converts each bookmaker's prices to no-margin probabilities and takes the median across available books. The BTTS request is per match and is limited to 10 matches per prediction run by default; set `FOOTBALL_ODDS_MAX_BTTS_EVENTS` to change the cap or `0` to turn BTTS lookups off. Set `FOOTBALL_ODDS_REGIONS` to change the bookmaker region (defaults to `uk`). Missing quotes fall back to model estimates. On Streamlit Community Cloud, add `ODDS_API_KEY` to the app's Secrets; locally, set it as an environment variable. Do not put API keys in source files or CSVs.
