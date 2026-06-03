from quantlab.config import settings
from quantlab.logging_utils import setup_logging, get_logger
from quantlab.paths import ensure_project_dirs, PROJECT_ROOT, RAW_DATA_DIR, OUTPUT_DIR


def main() -> None:
    setup_logging(settings.log_level)
    logger = get_logger("smoke_test")

    ensure_project_dirs()

    logger.info("Smoke test starting")
    logger.info("Project root: %s", PROJECT_ROOT)
    logger.info("Raw data dir: %s", RAW_DATA_DIR)
    logger.info("Output dir: %s", OUTPUT_DIR)
    logger.info("App name: %s", settings.app_name)
    logger.info("Smoke test completed successfully")


if __name__ == "__main__":
    main()