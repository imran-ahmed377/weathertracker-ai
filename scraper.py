from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

API_BASE = "https://vpic.nhtsa.dot.gov/api/vehicles"
DEFAULT_MAX_MAKES = 15
DEFAULT_YEAR_COUNT = 3
OUTPUT_DIR = Path("output")
EXCEL_PATH = OUTPUT_DIR / "cars.xlsx"
CSV_PATH = OUTPUT_DIR / "cars.csv"


@dataclass(frozen=True)
class CarRecord:
    listing_id: str
    title: str
    make: str
    model: str
    year: int
    transmission: str | None
    fuel: str | None
    mileage: str | None
    price: str | None
    location: str | None
    source_url: str
    scraped_at: str


def build_session() -> requests.Session:
    retry = Retry(
        total=4,
        backoff_factor=1.0,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods={"GET"},
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)

    session = requests.Session()
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update(
        {"User-Agent": "autotracker-ai/1.0 (+github-actions)"})
    return session


def fetch_json(session: requests.Session, url: str, timeout_seconds: int = 30) -> list[dict[str, Any]]:
    response = session.get(url, timeout=timeout_seconds)
    response.raise_for_status()

    payload = response.json()
    if not isinstance(payload, dict) or "Results" not in payload:
        raise ValueError(f"Unexpected API response shape from: {url}")

    results = payload["Results"]
    if not isinstance(results, list):
        raise ValueError(f"Unexpected results value from: {url}")

    return results


def fetch_makes(session: requests.Session, max_makes: int) -> list[dict[str, Any]]:
    url = f"{API_BASE}/GetMakesForVehicleType/car?format=json"
    rows = fetch_json(session, url)

    filtered = [row for row in rows if row.get("MakeName")]
    filtered.sort(key=lambda row: str(row.get("MakeName", "")))
    return filtered[:max_makes]


def fetch_models_for_make_year(session: requests.Session, make_name: str, year: int) -> list[dict[str, Any]]:
    url = (
        f"{API_BASE}/GetModelsForMakeYear/make/{make_name}"
        f"/modelyear/{year}/vehicletype/car?format=json"
    )
    return fetch_json(session, url)


def build_records(session: requests.Session, max_makes: int, year_count: int) -> list[CarRecord]:
    now_iso = datetime.now(timezone.utc).isoformat()
    makes = fetch_makes(session, max_makes=max_makes)

    current_year = datetime.now(timezone.utc).year
    years = [current_year - offset for offset in range(year_count)]

    records: list[CarRecord] = []
    for make in makes:
        make_name = str(make.get("MakeName", "")).strip()
        make_id = str(make.get("MakeId", "")).strip()
        if not make_name:
            continue

        for year in years:
            models = fetch_models_for_make_year(session, make_name, year)
            for model in models:
                model_name = str(model.get("Model_Name", "")).strip()
                model_id = str(model.get("Model_ID", "")).strip()
                if not model_name:
                    continue

                listing_id = f"nhtsa-{make_id}-{model_id}-{year}"
                title = f"{year} {make_name} {model_name}"
                source_url = (
                    f"{API_BASE}/GetModelsForMakeYear/make/{make_name}"
                    f"/modelyear/{year}/vehicletype/car?format=json"
                )

                records.append(
                    CarRecord(
                        listing_id=listing_id,
                        title=title,
                        make=make_name,
                        model=model_name,
                        year=year,
                        transmission=None,
                        fuel=None,
                        mileage=None,
                        price=None,
                        location=None,
                        source_url=source_url,
                        scraped_at=now_iso,
                    )
                )

    return records


def to_dataframe(records: list[CarRecord]) -> pd.DataFrame:
    rows = [record.__dict__ for record in records]
    df = pd.DataFrame(rows)

    column_order = [
        "listing_id",
        "title",
        "make",
        "model",
        "year",
        "transmission",
        "fuel",
        "mileage",
        "price",
        "location",
        "source_url",
        "scraped_at",
    ]

    for column in column_order:
        if column not in df.columns:
            df[column] = None

    df = df[column_order]
    df = df.drop_duplicates(subset=["listing_id"]).sort_values(
        by=["year", "make", "model"], ascending=[False, True, True]
    )
    return df


def write_outputs(df: pd.DataFrame) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_excel(EXCEL_PATH, index=False)
    df.to_csv(CSV_PATH, index=False)


def main() -> int:
    max_makes = int(os.getenv("MAX_MAKES", str(DEFAULT_MAX_MAKES)))
    year_count = int(os.getenv("YEAR_COUNT", str(DEFAULT_YEAR_COUNT)))

    if max_makes < 1 or year_count < 1:
        print("MAX_MAKES and YEAR_COUNT must be positive integers.")
        return 2

    session = build_session()
    try:
        records = build_records(
            session, max_makes=max_makes, year_count=year_count)
    except requests.HTTPError as exc:
        status = exc.response.status_code if exc.response is not None else "unknown"
        print(
            f"HTTP error while calling NHTSA API: status={status}, detail={exc}")
        return 1
    except requests.RequestException as exc:
        print(f"Network error while calling NHTSA API: {exc}")
        return 1
    except Exception as exc:
        print(f"Unexpected failure while scraping data: {exc}")
        return 1

    if not records:
        print(
            "No records were returned from the API; aborting to avoid empty output commit.")
        return 1

    df = to_dataframe(records)
    if df.empty:
        print("Dataframe is empty after normalization; aborting.")
        return 1

    write_outputs(df)
    print(f"Wrote {len(df)} rows to {EXCEL_PATH} and {CSV_PATH}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
