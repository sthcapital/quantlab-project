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
    log_level: str = "INFO"
    project_root: str = str(PROJECT_ROOT)
    data_dir: str = str(DATA_DIR)
    raw_data_dir: str = str(RAW_DATA_DIR)
    processed_data_dir: str = str(PROCESSED_DATA_DIR)
    external_data_dir: str = str(EXTERNAL_DATA_DIR)
    output_dir: str = str(OUTPUT_DIR)


settings = Settings()