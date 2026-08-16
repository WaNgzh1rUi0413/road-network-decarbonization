"""Repository paths shared by command-line modules."""

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
NETWORK_DIR = REPO_ROOT / "network"
PROCESSED_DATA = DATA_DIR / "processed" / "city_features_emissions.xlsx"
OUTPUT_DIR = REPO_ROOT / "outputs"


def ensure_output(*parts: str) -> Path:
    path = OUTPUT_DIR.joinpath(*parts)
    path.mkdir(parents=True, exist_ok=True)
    return path
