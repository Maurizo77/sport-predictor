"""Optional football totals and BTTS odds from The Odds API."""

import json
import os
import re
import tempfile
import time
import unicodedata
from datetime import timedelta
from statistics import median

import pandas as pd
import requests


API_ROOT = "https://api.the-odds-api.com/v4/sports"
CACHE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cache", "football_market_odds.json")
CACHE_TTL_SECONDS = 2 * 60 * 60
LEAGUE_KEYS = {
    "E0": "soccer_epl",
    "E1": "soccer_efl_champ",
    "E2": "soccer_england_league1",
    "E3": "soccer_england_league2",
    "SC0": "soccer_spl",
    "D1": "soccer_germany_bundesliga",
    "I1": "soccer_italy_serie_a",
    "SP1": "soccer_spain_la_liga",
    "F1": "soccer_france_ligue_one",
    "N1": "soccer_netherlands_eredivisie",
    "B1": "soccer_belgium_first_div",
    "P1": "soccer_portugal_primeira_liga",
    "T1": "soccer_turkey_super_league",
    "G1": "soccer_greece_super_league",
}
ALIASES = {
    "man utd": "manchester united",
    "man united": "manchester united",
    "man city": "manchester city",
    "spurs": "tottenham hotspur",
    "wolves": "wolverhampton wanderers",
    "west brom": "west bromwich albion",
    "qpr": "queens park rangers",
    "sheff utd": "sheffield united",
    "sheff wed": "sheffield wednesday",
    "notts forest": "nottingham forest",
    "brighton": "brighton and hove albion",
    "newcastle": "newcastle united",
    "leeds": "leeds united",
    "stoke": "stoke city",
    "norwich": "norwich city",
    "coventry": "coventry city",
    "luton": "luton town",
    "derby": "derby county",
    "sunderland": "sunderland afc",
}


def _cached_get(url, params):
    """Cache public odds payloads briefly; never persist the API key."""
    os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
    cache_key = url + "?" + "&".join(
        f"{key}={value}" for key, value in sorted(params.items()) if key != "apiKey"
    )
    try:
        with open(CACHE_FILE, encoding="utf-8") as handle:
            cache = json.load(handle)
    except (OSError, ValueError):
        cache = {}
    entry = cache.get(cache_key, {})
    if time.time() - entry.get("saved_at", 0) < CACHE_TTL_SECONDS:
        return entry.get("status", 200), entry.get("data"), None, True

    response = requests.get(url, params=params, timeout=25)
    try:
        data = response.json()
    except ValueError:
        data = None
    remaining = response.headers.get("x-requests-remaining")
    if response.status_code == 200 and data is not None:
        now = time.time()
        cache = {
            key: value for key, value in cache.items()
            if now - value.get("saved_at", 0) < 7 * 24 * 60 * 60
        }
        cache[cache_key] = {"saved_at": now, "status": response.status_code, "data": data}
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=os.path.dirname(CACHE_FILE),
            suffix=".tmp", delete=False,
        ) as handle:
            json.dump(cache, handle)
            temporary = handle.name
        os.replace(temporary, CACHE_FILE)
    return response.status_code, data, remaining, False


def _team_key(value):
    value = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode()
    value = re.sub(r"[^a-z0-9 ]", " ", value.lower())
    value = " ".join(value.split())
    value = ALIASES.get(value, value)
    # Club suffixes are inconsistent across feeds and do not distinguish teams.
    return " ".join(word for word in value.split() if word not in {"fc", "afc", "cf"})


def _fair_probability(outcomes, names):
    prices = {}
    for outcome in outcomes:
        name = str(outcome.get("name", "")).strip().lower()
        try:
            price = float(outcome.get("price"))
        except (TypeError, ValueError):
            continue
        if name in names and price > 1:
            prices[name] = price
    if not all(name in prices for name in names):
        return None
    implied = {name: 1.0 / prices[name] for name in names}
    total = sum(implied.values())
    return {name: value / total for name, value in implied.items()}


def _consensus(bookmakers, market_key, outcome_names, point=None):
    probabilities = {name: [] for name in outcome_names}
    for bookmaker in bookmakers:
        for market in bookmaker.get("markets", []):
            if market.get("key") != market_key:
                continue
            outcomes = market.get("outcomes", [])
            if point is not None:
                matching = [
                    item for item in outcomes
                    if item.get("point") is not None
                    and abs(float(item["point"]) - point) < 0.01
                ]
                outcomes = matching
            fair = _fair_probability(outcomes, outcome_names)
            if fair:
                for name, probability in fair.items():
                    probabilities[name].append(probability)
            break
    if not probabilities or not all(probabilities.values()):
        return None
    return {name: median(values) for name, values in probabilities.items()}, min(
        len(values) for values in probabilities.values()
    )


def _match_event(row, events):
    home, away = _team_key(row.HOME), _team_key(row.AWAY)
    try:
        fixture_date = pd.Timestamp(row.DATE).date()
    except (TypeError, ValueError):
        fixture_date = None
    candidates = []
    for event in events:
        if _team_key(event.get("home_team", "")) != home:
            continue
        if _team_key(event.get("away_team", "")) != away:
            continue
        if fixture_date is not None:
            try:
                event_date = pd.to_datetime(event["commence_time"], utc=True).date()
            except (KeyError, TypeError, ValueError):
                event_date = None
            if event_date is not None and abs(event_date - fixture_date) > timedelta(days=1):
                continue
        candidates.append(event)
    return candidates[0] if len(candidates) == 1 else None


def fetch_football_market_odds(fixtures):
    """Return per-fixture no-vig totals/BTTS consensus where quotes exist.

    BTTS uses one event-level API request per match and is capped at 10 matches
    by default to keep API-credit use bounded. Set FOOTBALL_ODDS_MAX_BTTS_EVENTS
    to adjust that limit, or set it to 0 to skip BTTS market requests.
    """
    columns = [
        "P_OVER25_MARKET", "P_BTTS_MARKET", "OVER25_BOOKS", "BTTS_BOOKS",
        "OVER25_SOURCE", "BTTS_SOURCE",
    ]
    result = pd.DataFrame(index=fixtures.index, columns=columns)
    result["P_OVER25_MARKET"] = float("nan")
    result["P_BTTS_MARKET"] = float("nan")
    result["OVER25_BOOKS"] = 0
    result["BTTS_BOOKS"] = 0
    result["OVER25_SOURCE"] = ""
    result["BTTS_SOURCE"] = ""

    api_key = os.environ.get("ODDS_API_KEY")
    if not api_key or fixtures.empty:
        return result
    regions = os.environ.get("FOOTBALL_ODDS_REGIONS", "uk").strip() or "uk"
    try:
        max_btts_events = max(0, int(os.environ.get("FOOTBALL_ODDS_MAX_BTTS_EVENTS", "10")))
    except ValueError:
        max_btts_events = 10

    matched_events = []
    remaining = None
    leagues = fixtures["LEAGUE"].dropna().astype(str).unique()
    for league in leagues:
        sport_key = LEAGUE_KEYS.get(league)
        if not sport_key:
            continue
        try:
            status, payload, quota, _cached = _cached_get(
                f"{API_ROOT}/{sport_key}/odds",
                {"apiKey": api_key, "regions": regions, "markets": "totals", "oddsFormat": "decimal"},
            )
            remaining = quota or remaining
            if status in (401, 402, 429):
                print(f"Football odds lookup stopped (HTTP {status}; API limit or access restriction).")
                break
            if status != 200 or not isinstance(payload, list):
                print(f"Football totals odds unavailable for {league} (HTTP {status}).")
                continue
            events = payload
            subset = fixtures[fixtures["LEAGUE"].astype(str) == league]
            for row in subset.itertuples():
                event = _match_event(row, events)
                if event is None:
                    continue
                totals = _consensus(
                    event.get("bookmakers", []), "totals", ["over", "under"], point=2.5
                )
                if totals:
                    probabilities, book_count = totals
                    result.loc[row.Index, "P_OVER25_MARKET"] = probabilities["over"]
                    result.loc[row.Index, "OVER25_BOOKS"] = book_count
                    result.loc[row.Index, "OVER25_SOURCE"] = f"The Odds API ({regions}; {book_count} books)"
                matched_events.append((row.Index, sport_key, event))
        except (requests.RequestException, ValueError, TypeError) as error:
            print(f"Football totals odds unavailable for {league}: {type(error).__name__}.")

    try:
        credits_left = int(remaining) if remaining is not None else None
    except (TypeError, ValueError):
        credits_left = None
    if max_btts_events and matched_events and credits_left != 0:
        requested = 0
        for fixture_index, sport_key, event in matched_events[:max_btts_events]:
            event_id = event.get("id")
            if not event_id:
                continue
            try:
                status, payload, quota, cached = _cached_get(
                    f"{API_ROOT}/{sport_key}/events/{event_id}/odds",
                    {"apiKey": api_key, "regions": regions, "markets": "btts", "oddsFormat": "decimal"},
                )
                if not cached:
                    requested += 1
                remaining = quota or remaining
                if status in (401, 402, 429):
                    print(f"BTTS market lookups stopped (HTTP {status}; API limit or access restriction).")
                    break
                if status != 200 or not isinstance(payload, dict):
                    continue
                consensus = _consensus(
                    payload.get("bookmakers", []), "btts", ["yes", "no"]
                )
                if consensus:
                    probabilities, book_count = consensus
                    result.loc[fixture_index, "P_BTTS_MARKET"] = probabilities["yes"]
                    result.loc[fixture_index, "BTTS_BOOKS"] = book_count
                    result.loc[fixture_index, "BTTS_SOURCE"] = f"The Odds API ({regions}; {book_count} books)"
            except (requests.RequestException, ValueError, TypeError) as error:
                print(f"BTTS odds lookup failed for one match: {type(error).__name__}.")
        print(f"BTTS market lookups used: {requested} (limit {max_btts_events}).")
    elif matched_events and credits_left == 0:
        print("BTTS market lookups skipped because the API reported no credits remaining.")
    if remaining is not None:
        print("Odds API credits remaining:", remaining)
    return result
