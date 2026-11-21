"""Fetch the next 49-match coupon from AbleFAST into pool_list.txt."""

import os
import re
import tempfile
from datetime import date, datetime
from html.parser import HTMLParser
from urllib.parse import urljoin, urlparse
from zoneinfo import ZoneInfo

import requests


BASE = os.path.dirname(os.path.abspath(__file__))
INDEX_URL = "https://ablefast.com/fixtures"
OUTPUT = os.path.join(BASE, "pool_list.txt")
EXPECTED_FIXTURES = 49


class CouponLinks(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []
        self._href = None
        self._text = []

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            self._href = dict(attrs).get("href")
            self._text = []

    def handle_data(self, data):
        if self._href is not None:
            self._text.append(data)

    def handle_endtag(self, tag):
        if tag == "a" and self._href is not None:
            self.links.append((self._href, " ".join(self._text)))
            self._href = None
            self._text = []


class CouponRows(HTMLParser):
    def __init__(self):
        super().__init__()
        self.rows = []
        self._row = None
        self._cell = None

    def handle_starttag(self, tag, attrs):
        if tag == "tr":
            self._row = []
        elif tag in ("td", "th") and self._row is not None:
            self._cell = []

    def handle_data(self, data):
        if self._cell is not None:
            self._cell.append(data)

    def handle_endtag(self, tag):
        if tag in ("td", "th") and self._cell is not None:
            value = re.sub(r"\s+", " ", " ".join(self._cell)).strip()
            self._row.append(value)
            self._cell = None
        elif tag == "tr" and self._row is not None:
            self.rows.append(self._row)
            self._row = None


def download(url):
    response = requests.get(
        url,
        headers={"User-Agent": "Sports-Predictor/1.0 (weekly fixture refresh)"},
        timeout=30,
    )
    response.raise_for_status()
    return response.text


def next_coupon_url(index_html):
    parser = CouponLinks()
    parser.feed(index_html)
    today = datetime.now(ZoneInfo("Europe/London")).date()
    candidates = []
    for href, label in parser.links:
        path = urlparse(href).path.rstrip("/")
        match = re.fullmatch(r"/fixtures/(\d{4}-\d{2}-\d{2})", path)
        if not match:
            continue
        coupon_date = date.fromisoformat(match.group(1))
        if coupon_date > today:
            candidates.append((coupon_date, urljoin(INDEX_URL, href), label))
    if not candidates:
        raise RuntimeError("AbleFAST did not list a future dated coupon on its fixtures page.")
    coupon_date, url, label = min(candidates)
    return coupon_date, url, label


def parse_fixtures(html):
    parser = CouponRows()
    parser.feed(html)
    fixtures = []
    for cells in parser.rows:
        if len(cells) < 4 or not cells[0].strip().isdigit():
            continue
        home, away = cells[1].strip(), cells[3].strip()
        if home and away and home.lower() != "home" and away.lower() != "away":
            fixtures.append((home, away))
    return fixtures


def save_fixtures(fixtures):
    if len(fixtures) != EXPECTED_FIXTURES:
        raise RuntimeError(
            f"Expected {EXPECTED_FIXTURES} fixtures, but AbleFAST returned {len(fixtures)}. "
            "The existing pool list was left unchanged."
        )
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="\n", dir=BASE, suffix=".tmp", delete=False
    ) as temporary:
        temporary_path = temporary.name
        for home, away in fixtures:
            temporary.write(f"{home} v {away}\n")
    try:
        os.replace(temporary_path, OUTPUT)
    finally:
        if os.path.exists(temporary_path):
            os.remove(temporary_path)


def main():
    try:
        index_html = download(INDEX_URL)
        coupon_date, coupon_url, label = next_coupon_url(index_html)
        fixtures = parse_fixtures(download(coupon_url))
        save_fixtures(fixtures)
    except Exception as error:
        print("Could not refresh the weekly pool coupon:", type(error).__name__, str(error))
        raise SystemExit(1)

    print("Coupon:", label.strip() or coupon_date.isoformat())
    print("Date:", coupon_date.isoformat())
    print("Fixtures loaded:", len(fixtures))
    print("Saved to:", OUTPUT)


if __name__ == "__main__":
    main()
