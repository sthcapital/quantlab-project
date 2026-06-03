from quantlab.io import write_json, read_json, write_csv
from quantlab.paths import OUTPUT_DIR


def test_json_roundtrip():
    path = OUTPUT_DIR / "test_payload.json"
    payload = {"app": "quantlab", "ok": True}

    write_json(path, payload)
    loaded = read_json(path)

    assert loaded == payload


def test_write_csv():
    path = OUTPUT_DIR / "test_rows.csv"
    rows = [
        {"symbol": "AAPL", "close": 100.0},
        {"symbol": "MSFT", "close": 200.0},
    ]

    write_csv(path, rows)

    assert path.exists()