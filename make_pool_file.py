import difflib
import os
import re
import sys

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

LIST_FILE = "pool_list.txt"
OUT_FILE = "pool_fixtures.csv"

if not os.path.exists(LIST_FILE):
    print("Create", LIST_FILE, "first: paste your coupon there, one match per line (Home v Away).")
    raise SystemExit

history = pd.read_csv("football_games.csv", usecols=["HomeTeam", "AwayTeam"])
known = sorted(set(history["HomeTeam"].dropna()) | set(history["AwayTeam"].dropna()))


def norm(text):
    t = str(text).lower().replace("&", " and ")
    t = re.sub(r"[^a-z0-9 ]", " ", t)
    return " ".join(w for w in t.split() if w != "fc")


norm_to_known = {norm(k): k for k in known}

ALIASES = {
    "manchester united": "Man United",
    "man utd": "Man United",
    "aston v": "Aston Villa",
    "manchester city": "Man City",
    "crystal p": "Crystal Palace",
    "q p r": "QPR",
    "sheff utd": "Sheffield United",
    "sheffield utd": "Sheffield United",
    "sheff wed": "Sheffield Weds",
    "bristol c": "Bristol City",
    "a wimbledon": "AFC Wimbledon",
    "burton a": "Burton",
    "oxford utd": "Oxford",
    "leyton o": "Leyton Orient",
    "bradford c": "Bradford",
    "newport co": "Newport County",
    "salford c": "Salford",
    "fleetwood": "Fleetwood Town",
    "dundee utd": "Dundee United",
    "st j stone": "St Johnstone",
    "heart": "Hearts",
    "e frankfurt": "Ein Frankfurt",
    "b dortmund": "Dortmund",
    "b leverkusen": "Leverkusen",
    "deportivo lc": "La Coruna",
    "r sociedad": "Sociedad",
    "nottingham forest": "Nott'm Forest",
    "wolverhampton wanderers": "Wolves",
    "wolverhampton": "Wolves",
    "tottenham hotspur": "Tottenham",
    "newcastle united": "Newcastle",
    "west ham united": "West Ham",
    "brighton and hove albion": "Brighton",
    "brighton hove albion": "Brighton",
    "sheffield wednesday": "Sheffield Weds",
    "leicester city": "Leicester",
    "norwich city": "Norwich",
    "leeds united": "Leeds",
    "west bromwich albion": "West Brom",
    "queens park rangers": "QPR",
    "preston north end": "Preston",
    "stoke city": "Stoke",
    "cardiff city": "Cardiff",
    "swansea city": "Swansea",
    "hull city": "Hull",
    "coventry city": "Coventry",
    "birmingham city": "Birmingham",
    "derby county": "Derby",
    "oxford united": "Oxford",
    "plymouth argyle": "Plymouth",
    "luton town": "Luton",
    "blackburn rovers": "Blackburn",
    "ipswich town": "Ipswich",
    "wigan athletic": "Wigan",
    "bolton wanderers": "Bolton",
    "huddersfield town": "Huddersfield",
    "peterborough united": "Peterboro",
    "peterborough": "Peterboro",
    "doncaster rovers": "Doncaster",
    "rotherham united": "Rotherham",
    "lincoln city": "Lincoln",
    "milton keynes dons": "Milton Keynes Dons",
    "mk dons": "Milton Keynes Dons",
    "afc bournemouth": "Bournemouth",
    "sheffield utd": "Sheffield United",
    "oldham athletic": "Oldham",
    "charlton athletic": "Charlton",
    "cambridge united": "Cambridge",
    "colchester united": "Colchester",
    "bradford city": "Bradford",
    "exeter city": "Exeter",
    "northampton town": "Northampton",
    "swindon town": "Swindon",
    "wycombe wanderers": "Wycombe",
    "shrewsbury town": "Shrewsbury",
    "grimsby town": "Grimsby",
    "tranmere rovers": "Tranmere",
}
SEPARATOR = re.compile(r"\s+(?:v|vs|vs\.|-|–|—)\s+", re.IGNORECASE)


def resolve(name):
    n = norm(name)
    if n in norm_to_known:
        return norm_to_known[n], "exact", []
    target = ALIASES.get(n)
    if target and target in known:
        return target, "alias", []
    close = difflib.get_close_matches(n, list(norm_to_known), n=1, cutoff=0.8)
    if close:
        return norm_to_known[close[0]], "check", []
    options = difflib.get_close_matches(n, list(norm_to_known), n=3, cutoff=0.5)
    return None, "missing", [norm_to_known[o] for o in options]


def parse(line):
    line = re.sub(r"^\s*\d{1,2}[\.\):]?\s+(?=[A-Za-z])", "", line.strip())
    parts = SEPARATOR.split(line, maxsplit=1)
    if len(parts) != 2:
        return None
    home = parts[0].strip()
    tokens = parts[1].split()
    numbers = []
    while tokens and len(numbers) < 3 and re.fullmatch(r"\d+(?:\.\d+)?", tokens[-1]):
        numbers.insert(0, tokens.pop())
    if len(numbers) == 3:
        odds = [float(x) for x in numbers]
    else:
        tokens = tokens + numbers
        odds = [None, None, None]
    return home, " ".join(tokens).strip(), odds


rows = []
checks = []
missing = []
bad = []
with open(LIST_FILE, encoding="utf-8-sig") as handle:
    for raw in handle:
        if not raw.strip() or raw.strip().startswith("#"):
            continue
        parsed = parse(raw)
        if parsed is None:
            bad.append(raw.strip())
            continue
        home, away, odds = parsed
        results = []
        for name in (home, away):
            results.append(resolve(name) + (name,))
        if any(r[0] is None for r in results):
            for target, status, options, original in results:
                if target is None:
                    missing.append((original, options))
            continue
        for target, status, options, original in results:
            if status == "check":
                checks.append((original, target))
        rows.append({
            "HomeTeam": results[0][0], "AwayTeam": results[1][0],
            "OddsHome": odds[0], "OddsDraw": odds[1], "OddsAway": odds[2],
        })

pd.DataFrame(rows).to_csv(OUT_FILE, index=False)
print("Fixtures written to", OUT_FILE + ":", len(rows))
if checks:
    print()
    print("Please CHECK these guesses (we changed the spelling to the closest name):")
    for original, target in checks:
        print("  ", original, "->", target)
if missing:
    print()
    print("NOT FOUND (these fixtures were left out). Fix the spelling in", LIST_FILE, "and run again:")
    for original, options in missing:
        print("  ", original, "-> closest:", ", ".join(options) if options else "no close match")
if bad:
    print()
    print("Lines I could not read (use 'Home v Away'):")
    for line in bad:
        print("  ", line)

    
