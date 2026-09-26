"""Download the Dominica Sperm Whale Project (DSWP) coda dataset.

Source: Sharma et al. (2024), "Contextual and combinatorial structure in sperm
whale vocalisations", Nature Communications 15, 3617.
Repository: https://github.com/pratyushasharma/sw-combinatoriality
Archive:    https://doi.org/10.5281/zenodo.10817697  (CC BY 4.0)

The download is pinned to a specific commit so results stay reproducible.

Usage:
    python scripts/download_data.py
"""
from pathlib import Path
from urllib.request import urlretrieve

COMMIT = "7228c8eed2cc27ddd23b74c51aeccec9d762389e"
BASE = f"https://raw.githubusercontent.com/pratyushasharma/sw-combinatoriality/{COMMIT}/data"
FILES = ["DominicaCodas.csv", "sperm-whale-dialogues.csv"]

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        target = RAW_DIR / name
        if target.exists():
            print(f"[skip] {name} already exists")
            continue
        print(f"[get ] {name}")
        urlretrieve(f"{BASE}/{name}", target)
    print(f"Done. Files in {RAW_DIR}")


if __name__ == "__main__":
    main()
