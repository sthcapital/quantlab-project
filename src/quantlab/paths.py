from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
PACKAGE_DIR = SRC_DIR / "quantlab"
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
OUTPUT_DIR = PROJECT_ROOT / "output"
TESTS_DIR = PROJECT_ROOT / "tests"


def ensure_project_dirs() -> None:
    for path in [
        DATA_DIR,
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        EXTERNAL_DATA_DIR,
        NOTEBOOKS_DIR,
        SCRIPTS_DIR,
        OUTPUT_DIR,
        TESTS_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)