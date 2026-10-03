"""Paths and the city set."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
ARTIFACTS = ROOT / "artifacts"
RESULTS = ROOT / "results"

# Calendar bitsets cover this many nights from each snapshot's date (Inside Airbnb calendars run
# about 365 nights forward; 384 = 6 x 64 leaves headroom).
HORIZON_BITS = 384

# Latest snapshot per city, read from https://insideairbnb.com/get-the-data/ on 2026-10-03.
CITIES = {
    "seattle": {"path": "united-states/wa/seattle", "latest": "2026-06-15", "currency": "USD"},
    "san-diego": {"path": "united-states/ca/san-diego", "latest": "2026-06-27", "currency": "USD"},
    "san-francisco": {"path": "united-states/ca/san-francisco", "latest": "2026-06-14", "currency": "USD"},
    "los-angeles": {"path": "united-states/ca/los-angeles", "latest": "2026-06-15", "currency": "USD"},
    "new-york-city": {"path": "united-states/ny/new-york-city", "latest": "2026-06-14", "currency": "USD"},
    "boston": {"path": "united-states/ma/boston", "latest": "2026-06-15", "currency": "USD"},
    "chicago": {"path": "united-states/il/chicago", "latest": "2026-06-24", "currency": "USD"},
    "austin": {"path": "united-states/tx/austin", "latest": "2026-06-22", "currency": "USD"},
    "london": {"path": "united-kingdom/england/london", "latest": "2026-06-19", "currency": "GBP"},
    "paris": {"path": "france/ile-de-france/paris", "latest": "2026-06-16", "currency": "EUR"},
}
US_CITIES = [c for c, v in CITIES.items() if v["path"].startswith("united-states")]

# Average length of stay per city, from Inside Airbnb's data assumptions page: San Francisco uses
# Airbnb's reported 5.5 nights, every other city the 3-night default.
AVG_LOS = {c: (5.5 if c == "san-francisco" else 3.0) for c in CITIES}
