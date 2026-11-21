"""Helpers for displaying scheduled event times in Nigerian time (WAT)."""

from datetime import datetime, time, timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


NIGERIA_TIMEZONE = ZoneInfo("Africa/Lagos")
FOOTBALL_FEED_TIMEZONE = "Europe/London"
TIME_UNAVAILABLE = "Time unavailable"


def _format_wat(event_datetime):
    return event_datetime.astimezone(NIGERIA_TIMEZONE).strftime("%Y-%m-%d %H:%M WAT")


def nba_event_time_wat(game_time_utc):
    """Convert the NBA scoreboard's UTC start timestamp to a Nigerian datetime."""
    if not game_time_utc:
        return TIME_UNAVAILABLE
    try:
        value = str(game_time_utc).strip()
        if value.endswith("Z"):
            value = value[:-1] + "+00:00"
        event_datetime = datetime.fromisoformat(value)
        if event_datetime.tzinfo is None:
            event_datetime = event_datetime.replace(tzinfo=timezone.utc)
        return _format_wat(event_datetime)
    except (TypeError, ValueError, OverflowError):
        return TIME_UNAVAILABLE


def football_event_time_wat(match_date, kickoff_time, source_timezone=FOOTBALL_FEED_TIMEZONE):
    """Convert a fixture's local kickoff to WAT; the public feed uses UK time.

    Manual fixture files can provide an optional TimeZone column with an IANA
    zone name (for example, Europe/Paris) when their Time field uses a different
    local clock.
    """
    if not kickoff_time:
        return TIME_UNAVAILABLE
    try:
        kickoff = time.fromisoformat(str(kickoff_time).strip())
        date = match_date.date() if hasattr(match_date, "date") else match_date
        event_datetime = datetime.combine(date, kickoff)
        if event_datetime.tzinfo is None:
            timezone_name = str(source_timezone).strip()
            if not timezone_name or timezone_name.lower() in {"nan", "none"}:
                timezone_name = FOOTBALL_FEED_TIMEZONE
            event_datetime = event_datetime.replace(tzinfo=ZoneInfo(timezone_name))
        return _format_wat(event_datetime)
    except (TypeError, ValueError, OverflowError, ZoneInfoNotFoundError):
        return TIME_UNAVAILABLE
