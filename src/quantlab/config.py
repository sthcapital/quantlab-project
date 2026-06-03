import os
from dataclasses import dataclass

from quantlab.paths import (
    PROJECT_ROOT,
    DATA_DIR,
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    EXTERNAL_DATA_DIR,
    OUTPUT_DIR,
)


@dataclass(frozen=True)
class Settings:
    app_name: str = "quantlab"
    environment: str = os.getenv("QUANTLAB_ENV", "dev")
    log_level: str = os.getenv("QUANTLAB_LOG_LEVEL", "INFO")
    market_data_provider: str = os.getenv("MARKET_DATA_PROVIDER", "")
    market_data_api_key: str = os.getenv("MARKET_DATA_API_KEY", "")
    market_data_api_secret: str = os.getenv("MARKET_DATA_API_SECRET", "")
    market_data_base_url: str = os.getenv("MARKET_DATA_BASE_URL", "")
    project_root: str = str(PROJECT_ROOT)
    data_dir: str = str(DATA_DIR)
    raw_data_dir: str = str(RAW_DATA_DIR)
    processed_data_dir: str = str(PROCESSED_DATA_DIR)
    external_data_dir: str = str(EXTERNAL_DATA_DIR)
    output_dir: str = str(OUTPUT_DIR)


settings = Settings()