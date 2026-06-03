from quantlab.paths import PROJECT_ROOT, RAW_DATA_DIR, OUTPUT_DIR, ensure_project_dirs


def test_project_root_exists():
    assert PROJECT_ROOT.exists()


def test_ensure_project_dirs():
    ensure_project_dirs()
    assert RAW_DATA_DIR.exists()
    assert OUTPUT_DIR.exists()