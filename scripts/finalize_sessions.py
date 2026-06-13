"""
scripts/finalize_sessions.py — auto-finalize options sessions against the EOD
flat file.  Replaces the manual ``rescore_options_session.py --write`` step.

Two modes (see quantlab.options_finalize for the design):

    --session DATE   Finalize one session (default: today).  Called by the
                     evening scan as its first step — the same-day flat file is
                     usually not published yet, so the session is normally
                     recorded intraday/unfinalized and the scan proceeds.

    --sweep          Finalize every prior unfinalized session before today.
                     Called by the morning job — yesterday's file has reliably
                     landed by then, so this is the normal path to ``final``.

Exit code is 2 when a session hit a credential/permission failure (a 403 that a
prior-date probe could not attribute to non-publication) so a wrapper can alert;
0 otherwise (an unpublished same-day file is the expected, non-error case).

Usage:
    python scripts/finalize_sessions.py                 # finalize today
    python scripts/finalize_sessions.py --session 2026-06-12
    python scripts/finalize_sessions.py --sweep
"""

from __future__ import annotations

import sys
from argparse import ArgumentParser
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def main() -> None:
    parser = ArgumentParser(description="Finalize options sessions against the EOD flat file.")
    parser.add_argument("--session", default=None,
                        help="Session date to finalize (YYYY-MM-DD, default: today)")
    parser.add_argument("--sweep", action="store_true",
                        help="Finalize every prior unfinalized session before today")
    parser.add_argument("--percentile", type=float, default=None,
                        help="Cross-sectional gate percentile (default: scanner config)")
    args = parser.parse_args()

    from quantlab.options_finalize import (
        STATUS_CREDENTIAL_FAILURE,
        finalize_session,
        sweep_unfinalized,
    )

    if args.sweep:
        results = sweep_unfinalized(percentile=args.percentile)
        if not results:
            print("Finalization sweep: no unfinalized prior sessions.")
        else:
            print(f"Finalization sweep — {len(results)} prior session(s):")
            for r in results:
                print(f"  {r.summary()}")
    else:
        session = date.fromisoformat(args.session) if args.session else date.today()
        r = finalize_session(session, percentile=args.percentile)
        print(r.summary())
        results = [r]

    # Loud, machine-detectable exit on a real credential/permission failure so
    # the calling shell can surface it — an unpublished same-day file is not one.
    if any(r.status == STATUS_CREDENTIAL_FAILURE for r in results):
        print("ERROR: options finalization hit a credential/permission failure "
              "(403 not attributable to non-publication) — check POLYGON_S3 creds.",
              file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
