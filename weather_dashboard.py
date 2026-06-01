from __future__ import annotations

import html
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

FORECAST_API = "https://api.open-meteo.com/v1/forecast"
HISTORICAL_API = "https://archive-api.open-meteo.com/v1/archive"
OUTPUT_DIR = Path("output")
DOCS_DIR = Path("docs")
CHARTS_DIR = DOCS_DIR / "charts"
OBSERVATIONS_CSV = OUTPUT_DIR / "weather_observations.csv"
DAILY_CSV = OUTPUT_DIR / "weather_daily.csv"
DASHBOARD_HTML = DOCS_DIR / "index.html"

DEFAULT_PAST_DAYS = 30
DEFAULT_FORECAST_DAYS = 7
TREND_WINDOW_DAYS = 365


@dataclass(frozen=True)
class City:
    name: str
    latitude: float
    longitude: float
    timezone: str


CITIES: tuple[City, ...] = (
    City("Toronto", 43.6532, -79.3832, "America/Toronto"),
    City("Montreal", 45.5017, -73.5673, "America/Toronto"),
    City("Vancouver", 49.2827, -123.1207, "America/Vancouver"),
    City("Calgary", 51.0447, -114.0719, "America/Edmonton"),
    City("Edmonton", 53.5461, -113.4938, "America/Edmonton"),
    City("Ottawa", 45.4215, -75.6972, "America/Toronto"),
    City("Winnipeg", 49.8951, -97.1384, "America/Winnipeg"),
    City("Quebec City", 46.8139, -71.2080, "America/Toronto"),
)


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
        {"User-Agent": "weathertracker-ai/1.0 (+github-actions)"})
    return session


def fetch_city_weather(
    session: requests.Session,
    city: City,
    past_days: int,
    forecast_days: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    params: dict[str, Any] = {
        "latitude": city.latitude,
        "longitude": city.longitude,
        "timezone": city.timezone,
        "past_days": past_days,
        "forecast_days": forecast_days,
        "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,wind_speed_10m_max",
        "current": "temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m,weather_code",
    }
    response = session.get(FORECAST_API, params=params, timeout=30)
    response.raise_for_status()

    payload = response.json()
    daily = payload.get("daily")
    current = payload.get("current")
    if not isinstance(daily, dict) or not isinstance(current, dict):
        raise ValueError(f"Unexpected Open-Meteo payload for {city.name}")

    daily_df = pd.DataFrame(
        {
            "date": daily.get("time", []),
            "temp_max_c": daily.get("temperature_2m_max", []),
            "temp_min_c": daily.get("temperature_2m_min", []),
            "precip_mm": daily.get("precipitation_sum", []),
            "wind_max_kmh": daily.get("wind_speed_10m_max", []),
        }
    )
    daily_df["city"] = city.name
    daily_df["timezone"] = city.timezone
    daily_df["date"] = pd.to_datetime(daily_df["date"], errors="coerce")
    daily_df = daily_df.dropna(subset=["date"])
    return daily_df, current


def fetch_city_climate_trend(
    session: requests.Session,
    city: City,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    params: dict[str, Any] = {
        "latitude": city.latitude,
        "longitude": city.longitude,
        "timezone": city.timezone,
        "start_date": start_date,
        "end_date": end_date,
        "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,wind_speed_10m_max",
    }
    response = session.get(HISTORICAL_API, params=params, timeout=30)
    response.raise_for_status()

    payload = response.json()
    daily = payload.get("daily")
    if not isinstance(daily, dict):
        raise ValueError(
            f"Unexpected Open-Meteo historical payload for {city.name}")

    daily_df = pd.DataFrame(
        {
            "date": daily.get("time", []),
            "temp_max_c": daily.get("temperature_2m_max", []),
            "temp_min_c": daily.get("temperature_2m_min", []),
            "precip_mm": daily.get("precipitation_sum", []),
            "wind_max_kmh": daily.get("wind_speed_10m_max", []),
        }
    )
    daily_df["city"] = city.name
    daily_df["timezone"] = city.timezone
    daily_df["date"] = pd.to_datetime(daily_df["date"], errors="coerce")
    daily_df = daily_df.dropna(subset=["date"])
    return daily_df


def build_observation_row(city: City, current: dict[str, Any], run_ts: str) -> dict[str, Any]:
    return {
        "scraped_at_utc": run_ts,
        "city": city.name,
        "timezone": city.timezone,
        "temp_c": current.get("temperature_2m"),
        "humidity_pct": current.get("relative_humidity_2m"),
        "precip_mm": current.get("precipitation"),
        "wind_kmh": current.get("wind_speed_10m"),
        "weather_code": current.get("weather_code"),
    }


def write_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def append_observations(observation_df: pd.DataFrame) -> pd.DataFrame:
    allowed_cities = {city.name for city in CITIES}

    if OBSERVATIONS_CSV.exists():
        existing = pd.read_csv(OBSERVATIONS_CSV)
        combined = pd.concat([existing, observation_df], ignore_index=True)
    else:
        combined = observation_df.copy()

    combined = combined[combined["city"].isin(allowed_cities)]

    combined = combined.drop_duplicates(
        subset=["scraped_at_utc", "city"], keep="last")
    combined["scraped_at_utc"] = pd.to_datetime(
        combined["scraped_at_utc"], errors="coerce")
    combined = combined.dropna(subset=["scraped_at_utc"]).sort_values([
        "city", "scraped_at_utc"])
    combined["scraped_at_utc"] = combined["scraped_at_utc"].dt.strftime(
        "%Y-%m-%dT%H:%M:%SZ")
    write_csv(OBSERVATIONS_CSV, combined)
    return combined


def slugify(name: str) -> str:
    return name.lower().replace(" ", "-")


def generate_city_chart(city_daily: pd.DataFrame, city_name: str) -> str:
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    chart_file = f"{slugify(city_name)}-trend.png"
    chart_path = CHARTS_DIR / chart_file

    city_daily = city_daily.sort_values("date")
    fig, (ax_temp, ax_rain) = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(12, 8),
        sharex=True,
        gridspec_kw={"height_ratios": [2.4, 1.3]},
    )

    ax_temp.plot(city_daily["date"], city_daily["temp_max_c"],
                 color="#f24822", linewidth=2.2, label="Daily max")
    ax_temp.plot(city_daily["date"], city_daily["temp_min_c"],
                 color="#228be6", linewidth=2.2, label="Daily min")
    ax_temp.set_ylabel("Temp (degC)")
    ax_temp.set_title(f"{city_name} climate trend", fontsize=14, pad=10)
    ax_temp.grid(alpha=0.25, linestyle="--")
    ax_temp.legend(loc="upper right")

    ax_rain.bar(city_daily["date"], city_daily["precip_mm"],
                color="#16a34a", width=0.85)
    ax_rain.set_ylabel("Rain (mm)")
    ax_rain.grid(alpha=0.2, linestyle="--", axis="y")

    fig.autofmt_xdate(rotation=35)
    fig.tight_layout()
    fig.savefig(chart_path, dpi=160)
    plt.close(fig)
    return f"charts/{chart_file}"


def build_dashboard_html(
    observations: pd.DataFrame,
    daily_df: pd.DataFrame,
    chart_paths: dict[str, str],
    generated_at: str,
) -> str:
    latest = observations.sort_values("scraped_at_utc").groupby(
        "city", as_index=False).tail(1)
    latest = latest.sort_values("city")

    cards: list[str] = []
    for _, row in latest.iterrows():
        cards.append(
            """
            <article class="card">
              <h3>{city}</h3>
              <p class="temp">{temp} degC</p>
              <p>Humidity: {humidity}% | Wind: {wind} km/h</p>
              <p>Rain now: {rain} mm</p>
              <p class="meta">Last update: {updated}</p>
            </article>
            """.format(
                city=html.escape(str(row["city"])),
                temp=row.get("temp_c", "n/a"),
                humidity=row.get("humidity_pct", "n/a"),
                wind=row.get("wind_kmh", "n/a"),
                rain=row.get("precip_mm", "n/a"),
                updated=html.escape(str(row["scraped_at_utc"])),
            )
        )

    chart_blocks: list[str] = []
    for city, rel_path in sorted(chart_paths.items()):
        city_rows = daily_df[daily_df["city"]
                             == city].copy().sort_values("date")
        if city_rows.empty:
            continue
        window_start = city_rows["date"].min().strftime("%Y-%m-%d")
        window_end = city_rows["date"].max().strftime("%Y-%m-%d")
        chart_blocks.append(
            """
            <section class="chart-card">
              <div class="chart-head">
                <h3>{city}</h3>
                <p>{window_start} to {window_end}</p>
              </div>
              <img src="{img}" alt="{city} climate trend chart" loading="lazy" />
            </section>
            """.format(
                city=html.escape(city),
                window_start=window_start,
                window_end=window_end,
                img=html.escape(rel_path),
            )
        )

    return """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Weather and Environment Dashboard</title>
  <style>
    :root {{
      --ink: #0f172a;
      --muted: #334155;
      --panel: #f8fafc;
      --line: #dbeafe;
      --accent: #065f46;
      --bg1: #dbeafe;
      --bg2: #f0fdf4;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      color: var(--ink);
      font-family: "Segoe UI", "Aptos", Tahoma, sans-serif;
      background:
        radial-gradient(1200px 500px at 10% -10%, #bae6fd 0, transparent 65%),
        radial-gradient(1000px 500px at 90% 0%, #bbf7d0 0, transparent 70%),
        linear-gradient(165deg, var(--bg1), var(--bg2));
      min-height: 100vh;
    }}
    .container {{
      width: min(1150px, calc(100% - 2rem));
      margin: 2rem auto 3rem;
    }}
    .hero {{
      background: rgba(255, 255, 255, 0.8);
      border: 1px solid var(--line);
      border-radius: 18px;
      padding: 1.2rem 1.4rem;
      backdrop-filter: blur(8px);
    }}
    h1 {{ margin: 0 0 0.2rem; }}
    .hero p {{ margin: 0.2rem 0; color: var(--muted); }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
      gap: 0.9rem;
      margin-top: 1rem;
    }}
    .card {{
      background: var(--panel);
      border-radius: 14px;
      border: 1px solid var(--line);
      padding: 0.9rem;
      box-shadow: 0 10px 30px rgba(15, 23, 42, 0.08);
    }}
    .card h3 {{ margin: 0 0 0.2rem; }}
    .temp {{ margin: 0.2rem 0 0.4rem; font-size: 1.5rem; font-weight: 700; color: var(--accent); }}
    .meta {{ color: #475569; font-size: 0.9rem; }}
    .charts {{
      margin-top: 1.3rem;
      display: grid;
      grid-template-columns: 1fr;
      gap: 1rem;
    }}
    .chart-card {{
      background: rgba(255, 255, 255, 0.88);
      border: 1px solid var(--line);
      border-radius: 16px;
      padding: 1rem;
    }}
    .chart-head {{
      display: flex;
      justify-content: space-between;
      align-items: baseline;
      gap: 0.8rem;
      margin-bottom: 0.5rem;
    }}
    .chart-head h3 {{ margin: 0; }}
    .chart-head p {{ margin: 0; color: var(--muted); font-size: 0.9rem; }}
    img {{ width: 100%; height: auto; border-radius: 10px; border: 1px solid #bfdbfe; }}
    @media (max-width: 700px) {{
      .container {{ width: min(1150px, calc(100% - 1rem)); }}
      .hero {{ padding: 0.9rem; }}
      h1 {{ font-size: 1.45rem; }}
    }}
  </style>
</head>
<body>
  <main class="container">
    <section class="hero">
      <h1>Weather and Environment Dashboard</h1>
      <p>Daily refresh powered by Open-Meteo. Includes recent climate trends and short-range forecast.</p>
      <p>Generated at: {generated_at}</p>
    </section>

    <section class="grid">
      {cards}
    </section>

    <section class="charts">
      {charts}
    </section>
  </main>
</body>
</html>
    """.format(generated_at=html.escape(generated_at), cards="\n".join(cards), charts="\n".join(chart_blocks))


def run() -> int:
    past_days = int(os.getenv("PAST_DAYS", str(DEFAULT_PAST_DAYS)))
    forecast_days = int(os.getenv("FORECAST_DAYS", str(DEFAULT_FORECAST_DAYS)))
    if past_days < 1 or forecast_days < 1:
        print("PAST_DAYS and FORECAST_DAYS must be positive integers.")
        return 2

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    (DOCS_DIR / ".nojekyll").write_text("", encoding="utf-8")

    for chart_file in CHARTS_DIR.glob("*-trend.png"):
        chart_file.unlink(missing_ok=True)

    run_ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    trend_end_date = (datetime.now(timezone.utc).date() - timedelta(days=1))
    trend_start_date = trend_end_date - timedelta(days=TREND_WINDOW_DAYS - 1)
    session = build_session()

    daily_frames: list[pd.DataFrame] = []
    observation_rows: list[dict[str, Any]] = []
    chart_paths: dict[str, str] = {}

    for city in CITIES:
        daily_df, current = fetch_city_weather(
            session, city, past_days=past_days, forecast_days=forecast_days)
        daily_frames.append(daily_df)
        observation_rows.append(build_observation_row(city, current, run_ts))
        trend_df = fetch_city_climate_trend(
            session,
            city,
            start_date=trend_start_date.strftime("%Y-%m-%d"),
            end_date=trend_end_date.strftime("%Y-%m-%d"),
        )
        chart_paths[city.name] = generate_city_chart(trend_df, city.name)

    all_daily = pd.concat(daily_frames, ignore_index=True)
    all_daily = all_daily.sort_values(
        ["city", "date"]) if not all_daily.empty else all_daily
    all_daily_to_write = all_daily.copy()
    if "date" in all_daily_to_write.columns:
        all_daily_to_write["date"] = all_daily_to_write["date"].dt.strftime(
            "%Y-%m-%d")
    write_csv(DAILY_CSV, all_daily_to_write)

    observations = pd.DataFrame(observation_rows)
    all_observations = append_observations(observations)

    dashboard = build_dashboard_html(
        all_observations, all_daily, chart_paths, run_ts)
    DASHBOARD_HTML.write_text(dashboard, encoding="utf-8")

    print(f"Wrote {len(all_daily_to_write)} daily weather rows to {DAILY_CSV}.")
    print(
        f"Wrote {len(all_observations)} observation rows to {OBSERVATIONS_CSV}.")
    print(f"Dashboard generated at {DASHBOARD_HTML}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
