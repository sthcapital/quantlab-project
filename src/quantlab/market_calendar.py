"""
quantlab.market_calendar — DST-aware US market hours and cron schedule builder.

Converts New York local times to UTC, accounting for EDT/EST transitions.
Uses pytz with America/New_York so every conversion is unambiguous even on
DST transition days (pytz handles the spring-forward / fall-back hour correctly).

DST schedule (US, second Sunday of March → first Sunday of November):
    2026: Spring March  8  |  Fall November  1
    2027: Spring March 14  |  Fall November  7
    2028: Spring March 13  |  Fall November  5

Market session times (NY local):
    Pre-market scan target : 09:00 AM  (run before the 09:30 open)
    Regular session open   : 09:30 AM
    Regular session close  : 04:00 PM
    EOD return tracker     : 04:30 PM  (30 min after close, prices settled)
"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta
from typing import NamedTuple

import pytz

NY_TZ = pytz.timezone("America/New_York")
UTC   = pytz.UTC

# ── Holiday computation helpers ────────────────────────────────────────────────

def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    """Return the nth occurrence of weekday (Mon=0…Sun=6) in year/month."""
    first = date(year, month, 1)
    delta = (weekday - first.weekday()) % 7
    return first + timedelta(days=delta + 7 * (n - 1))


def _last_weekday(year: int, month: int, weekday: int) -> date:
    """Return the last occurrence of weekday in year/month."""
    if month == 12:
        last = date(year + 1, 1, 1) - timedelta(days=1)
    else:
        last = date(year, month + 1, 1) - timedelta(days=1)
    return last - timedelta(days=(last.weekday() - weekday) % 7)


def _easter(year: int) -> date:
    """
    Compute Easter Sunday (Western/Gregorian) using the Meeus/Jones/Butcher
    algorithm.  Returns the ecclesiastically correct date.

    Verified against authoritative sources for 2026–2028:
        2026-04-05  2027-03-28  2028-04-16
    """
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month, p = divmod(h + l - 7 * m + 114, 31)
    return date(year, month, p + 1)


def _observed(d: date) -> date:
    """NYSE holiday observation rule: Sat → Fri, Sun → Mon."""
    if d.weekday() == 5:
        return d - timedelta(days=1)
    if d.weekday() == 6:
        return d + timedelta(days=1)
    return d


def _nyse_holidays(year: int) -> frozenset[date]:
    """
    Return the complete set of NYSE market holidays for one calendar year.

    NYSE observes these ten holidays:
        New Year's Day, MLK Jr. Day, Presidents Day, Good Friday,
        Memorial Day, Juneteenth, Independence Day, Labor Day,
        Thanksgiving, Christmas Day.
    """
    holidays: set[date] = set()

    # Fixed-date holidays with Saturday→Friday / Sunday→Monday observation
    for mo, da in [(1, 1), (6, 19), (7, 4), (12, 25)]:
        holidays.add(_observed(date(year, mo, da)))

    # Floating weekday holidays
    holidays.add(_nth_weekday(year, 1, 0, 3))   # MLK Day: 3rd Mon of Jan
    holidays.add(_nth_weekday(year, 2, 0, 3))   # Presidents Day: 3rd Mon of Feb
    holidays.add(_last_weekday(year, 5, 0))      # Memorial Day: last Mon of May
    holidays.add(_nth_weekday(year, 9, 0, 1))   # Labor Day: 1st Mon of Sep
    holidays.add(_nth_weekday(year, 11, 3, 4))  # Thanksgiving: 4th Thu of Nov

    # Good Friday: always the Friday before Easter (no observation needed)
    holidays.add(_easter(year) - timedelta(days=2))

    return frozenset(holidays)


# ── Public holiday API ────────────────────────────────────────────────────────

#: Precomputed NYSE holiday set covering 2026–2028.
#: Used by is_market_open() for O(1) lookups.
US_MARKET_HOLIDAYS: frozenset[date] = frozenset(
    h for yr in range(2026, 2029) for h in _nyse_holidays(yr)
)


def prev_trading_day(d: date) -> date:
    """Return the most recent NYSE trading day strictly before d."""
    d = d - timedelta(days=1)
    while not is_market_open(d):
        d = d - timedelta(days=1)
    return d


def next_trading_day(d: date) -> date:
    """Return the next NYSE trading day strictly after d."""
    d = d + timedelta(days=1)
    while not is_market_open(d):
        d = d + timedelta(days=1)
    return d


def is_market_open(dt: date) -> bool:
    """
    Return True when the NYSE is open for regular trading on dt.

    Returns False on weekends and on NYSE-observed holidays for 2026–2028.
    Holidays outside that window default to weekday-only check (no holiday
    database loaded for other years).

    Args:
        dt: Calendar date to check.

    Returns:
        bool — True means the exchange is expected to be open.
    """
    if dt.weekday() >= 5:   # Saturday=5, Sunday=6
        return False
    return dt not in US_MARKET_HOLIDAYS

# ── DST transition calendar ────────────────────────────────────────────────────
# US DST: clocks spring forward (2nd Sun of March, 2 AM → 3 AM EDT)
#         clocks fall back     (1st Sun of November, 2 AM → 1 AM EST)
# Pre-computed through 2028 so update_crontab.sh works without network access.

DST_TRANSITIONS: list[tuple[str, date, str]] = [
    ("spring_2026", date(2026,  3,  8), "2nd Sunday of March  → EDT (UTC-4)"),
    ("fall_2026",   date(2026, 11,  1), "1st Sunday of November → EST (UTC-5)"),
    ("spring_2027", date(2027,  3, 14), "2nd Sunday of March  → EDT (UTC-4)"),
    ("fall_2027",   date(2027, 11,  7), "1st Sunday of November → EST (UTC-5)"),
    ("spring_2028", date(2028,  3, 13), "2nd Sunday of March  → EDT (UTC-4)"),
    ("fall_2028",   date(2028, 11,  5), "1st Sunday of November → EST (UTC-5)"),
]

# Market session times (New York local, no tzinfo — applied to a specific date below)
SCAN_LOCAL       = time(9,  0)   # pre-market scan trigger
MARKET_OPEN_LOCAL = time(9, 30)   # regular session open
MARKET_CLOSE_LOCAL = time(16, 0)  # regular session close
EOD_TRACK_LOCAL  = time(16, 30)   # post-close return tracker


# ── DST detection ─────────────────────────────────────────────────────────────

def is_dst(dt: date | None = None) -> bool:
    """
    Return True when New York is on daylight saving time (EDT, UTC-4) for dt.
    Return False when on standard time (EST, UTC-5).

    Uses noon local time so the check is unambiguous on transition days
    (the transition occurs at 2 AM; by noon the offset is already settled).

    Args:
        dt: Calendar date.  Defaults to today.
    """
    dt = dt or date.today()
    aware = NY_TZ.localize(datetime(dt.year, dt.month, dt.day, 12, 0, 0))
    return bool(aware.dst())


def utc_offset_hours(dt: date | None = None) -> int:
    """
    Return the UTC offset in whole hours for New York on dt.

    Returns -4 (EDT) or -5 (EST).

    Args:
        dt: Calendar date.  Defaults to today.
    """
    dt = dt or date.today()
    aware = NY_TZ.localize(datetime(dt.year, dt.month, dt.day, 12, 0, 0))
    return int(aware.utcoffset().total_seconds() / 3600)


# ── UTC conversion ─────────────────────────────────────────────────────────────

class UtcTime(NamedTuple):
    """A UTC hour:minute pair with cron and display helpers."""

    hour: int
    minute: int

    def cron_fields(self) -> str:
        """Return 'MINUTE HOUR' cron fields (leftmost two fields)."""
        return f"{self.minute} {self.hour}"

    def __str__(self) -> str:
        return f"{self.hour:02d}:{self.minute:02d} UTC"


def to_utc(local_time: time, dt: date) -> UtcTime:
    """
    Convert a New York wall-clock time on dt to UTC.

    Args:
        local_time: New York local time (e.g. ``time(9, 0)`` for 9:00 AM ET).
        dt:         Calendar date; determines EDT vs EST offset.

    Returns:
        UtcTime with the corresponding UTC hour and minute.

    Example::

        to_utc(time(9, 0), date(2026, 6, 4))   # → UtcTime(13, 0)  (EDT)
        to_utc(time(9, 0), date(2026, 1, 15))  # → UtcTime(14, 0)  (EST)
    """
    local_dt = NY_TZ.localize(
        datetime(dt.year, dt.month, dt.day, local_time.hour, local_time.minute)
    )
    utc_dt = local_dt.astimezone(UTC)
    return UtcTime(utc_dt.hour, utc_dt.minute)


def get_market_open_utc(dt: date | None = None) -> UtcTime:
    """Return UTC time of NY market open (09:30 AM ET) for dt (default today)."""
    return to_utc(MARKET_OPEN_LOCAL, dt or date.today())


def get_scan_utc(dt: date | None = None) -> UtcTime:
    """Return UTC time of the pre-market scan target (09:00 AM ET) for dt."""
    return to_utc(SCAN_LOCAL, dt or date.today())


def get_eod_utc(dt: date | None = None) -> UtcTime:
    """Return UTC time of the EOD return tracker (04:30 PM ET) for dt."""
    return to_utc(EOD_TRACK_LOCAL, dt or date.today())


# ── Cron schedule builder ──────────────────────────────────────────────────────

def cron_schedule_for_date(dt: date | None = None) -> dict[str, str]:
    """
    Return DST-correct cron fields for the QuantLab automated schedule.

    Call this on any date to get the right UTC times for the crontab.
    The returned dict is consumed by ``update_crontab.sh``.

    Args:
        dt: Date to compute for (default: today).

    Returns:
        Dict with keys:

        ``scan_cron``   — ``"MINUTE HOUR"`` fields for the morning scan
        ``eod_cron``    — ``"MINUTE HOUR"`` fields for the EOD tracker
        ``tz_name``     — ``"EDT"`` or ``"EST"``
        ``utc_offset``  — ``"-4"`` or ``"-5"``
        ``scan_utc``    — human label e.g. ``"13:00 UTC"``
        ``eod_utc``     — human label e.g. ``"20:30 UTC"``

    Example (during EDT, UTC-4)::

        {'scan_cron': '0 13', 'eod_cron': '30 20',
         'tz_name': 'EDT', 'utc_offset': '-4',
         'scan_utc': '13:00 UTC', 'eod_utc': '20:30 UTC'}
    """
    dt      = dt or date.today()
    scan    = get_scan_utc(dt)
    eod     = get_eod_utc(dt)
    offset  = utc_offset_hours(dt)
    tz_name = "EDT" if is_dst(dt) else "EST"

    return {
        "scan_cron":  scan.cron_fields(),
        "eod_cron":   eod.cron_fields(),
        "tz_name":    tz_name,
        "utc_offset": str(offset),
        "scan_utc":   str(scan),
        "eod_utc":    str(eod),
    }
