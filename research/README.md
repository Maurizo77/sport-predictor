# Research scripts

These scripts are optional experiments and diagnostics. Run them from the project root, for example:

```powershell
python research/backtests/model.py
python research/diagnostics/today_games.py
```

The scripts read the root-level CSV files. They are kept out of the main project folder so the dashboard and its data-update workflow are easier to find.

- `backtests/`: NBA and football model experiments, backtests, and total-points analysis.
- `diagnostics/`: feature inspection, NBA scoreboard inspection, and a one-season NBA data download utility.
- `legacy/ake_pool_file_duplicate.py`: duplicate copy of the supported root-level `make_pool_file.py`; retained for reference and not part of the normal workflow.
