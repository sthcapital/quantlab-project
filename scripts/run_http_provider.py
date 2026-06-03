from datetime import date

from quantlab.logging_utils import get_logger, setup_logging
from quantlab.providers.http import HttpMarketDataProvider


def main() -> None:
    setup_logging("INFO")
    logger = get_logger("run_http_provider")

    provider = HttpMarketDataProvider()
    bars = provider.get_daily_bars(
        symbol="AAPL",
        start_date=date(2026, 1, 2),
        end_date=date(2026, 1, 9),
    )

    logger.info("Fetched %s bars", len(bars))
    if bars:
        logger.info("First bar: %s", bars[0])
        logger.info("Last bar: %s", bars[-1])


if __name__ == "__main__":
    main()