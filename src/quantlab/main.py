from pathlib import Path
import sys

def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    print("quantlab package is working")
    print(f"Python: {sys.version.split()[0]}")
    print(f"Project root: {project_root}")

if __name__ == "__main__":
    main()