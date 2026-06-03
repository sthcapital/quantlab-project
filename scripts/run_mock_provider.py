
from datetime import date

from quantlab.io import output_path, write_csv, write_json
from quantlab.logging_utils import get_logger, setup_logging
from quantlab.providers.mock import MockMarketDataProvider
from quantlab.services.market_data_service import MarketDataService


def main() -> None:
    setup_logging("INFO")
    logger = get_logger("run_mock_provider")

    provider = MockMarketDataProvider()
    service = MarketDataService(provider)

    symbol = "AAPL"
    start_date = date(2026, 1, 1)
    end_date = date(2026, 1, 5)

    rows = service.get_daily_bars_as_rows(
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
    )

    csv_path = output_path("mock_daily_bars.csv")
    json_path = output_path("mock_daily_bars_summary.json")

    write_csv(csv_path, rows)
    write_json(
        json_path,
        {
            "symbol": symbol,
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "row_count": len(rows),
            "output_csv": str(csv_path),
        },
    )

    logger.info("Wrote %s rows for %s", len(rows), symbol)
    logger.info("CSV output: %s", csv_path)
    logger.info("JSON output: %s", json_path)


if __name__ == "__main__":
    main()