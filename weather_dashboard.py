from __future__ import annotations

import json
import html
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

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
        "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,rain_sum,snowfall_sum,wind_speed_10m_max",
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
            "rain_mm": daily.get("rain_sum", []),
            "snowfall_cm": daily.get("snowfall_sum", []),
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


def _js_value(value: Any) -> Any:
    if pd.isna(value):
        return None
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")
    return value


def build_city_chart_section(city_daily: pd.DataFrame, city_name: str) -> str:
    city_daily = city_daily.sort_values("date").copy()
    city_daily["date_label"] = city_daily["date"].dt.strftime("%Y-%m-%d")
    section_id = f"{slugify(city_name)}-section"
    chart_id = f"{slugify(city_name)}-chart"
    precip_chart_id = f"{slugify(city_name)}-precip-chart"
    detail_id = f"{slugify(city_name)}-detail"
    rain_id = f"{slugify(city_name)}-rain"
    snowfall_id = f"{slugify(city_name)}-snowfall"

    point_rows: list[dict[str, Any]] = []
    for _, row in city_daily.iterrows():
        point_rows.append(
            {
                "date": _js_value(row["date_label"]),
                "temp_max_c": _js_value(row["temp_max_c"]),
                "temp_min_c": _js_value(row["temp_min_c"]),
                "precip_mm": _js_value(row["precip_mm"]),
                "rain_mm": _js_value(row["rain_mm"]),
                "snowfall_cm": _js_value(row["snowfall_cm"]),
                "wind_max_kmh": _js_value(row["wind_max_kmh"]),
            }
        )

    data_json = json.dumps(point_rows)
    title_json = json.dumps(city_name)
    chart_id_json = json.dumps(chart_id)
    precip_chart_id_json = json.dumps(precip_chart_id)
    detail_id_json = json.dumps(detail_id)

    return f"""
        <section id="{section_id}" class="chart-card" tabindex="-1">
            <div class="chart-head">
                <h3>{html.escape(city_name)}</h3>
                <p>Click any point to inspect that day.</p>
            </div>
            <div id="{chart_id}" class="interactive-chart"></div>
            <div class="chart-head chart-head-secondary">
                <h3>Rain and snowfall</h3>
                <p>Separate trace view with the same click details.</p>
            </div>
            <div id="{precip_chart_id}" class="interactive-chart"></div>
            <div id="{detail_id}" class="day-details"></div>
            <div class="metric-panels">
                <div class="metric-panel">
                    <span>Rain</span>
                    <strong id="{rain_id}" class="metric-value">n/a</strong>
                    <p class="metric-note">Daily rain accumulation for the selected day.</p>
                </div>
                <div class="metric-panel">
                    <span>Snowfall</span>
                    <strong id="{snowfall_id}" class="metric-value">n/a</strong>
                    <p class="metric-note">Daily snowfall accumulation for the selected day.</p>
                </div>
            </div>
            <script>
                (function() {{
                    const chartId = {chart_id_json};
                    const precipChartId = {precip_chart_id_json};
                    const detailId = {detail_id_json};
                    const rainId = {json.dumps(rain_id)};
                    const snowfallId = {json.dumps(snowfall_id)};
                    const cityName = {title_json};
                    const rows = {data_json};
                    const chartEl = document.getElementById(chartId);
                    const precipChartEl = document.getElementById(precipChartId);
                    const detailEl = document.getElementById(detailId);
                    const rainEl = document.getElementById(rainId);
                    const snowfallEl = document.getElementById(snowfallId);

                    function formatValue(value, unit) {{
                        if (value === null || value === undefined || Number.isNaN(value)) {{
                            return 'n/a';
                        }}
                        return `${{value}} ${{unit}}`;
                    }}

                    function renderDetails(index) {{
                        const row = rows[index];
                        if (!row) {{
                            detailEl.innerHTML = '<p>No daily data available.</p>';
                            rainEl.textContent = 'n/a';
                            snowfallEl.textContent = 'n/a';
                            return;
                        }}
                        rainEl.textContent = formatValue(row.rain_mm, 'mm');
                        snowfallEl.textContent = formatValue(row.snowfall_cm, 'cm');
                        detailEl.innerHTML = `
                            <div class="detail-grid">
                                <div><span>Date</span><strong>${{row.date}}</strong></div>
                                <div><span>Max Temp</span><strong>${{row.temp_max_c ?? 'n/a'}} °C</strong></div>
                                <div><span>Min Temp</span><strong>${{row.temp_min_c ?? 'n/a'}} °C</strong></div>
                                <div><span>Precipitation</span><strong>${{row.precip_mm ?? 'n/a'}} mm</strong></div>
                                <div><span>Wind Max</span><strong>${{row.wind_max_kmh ?? 'n/a'}} km/h</strong></div>
                            </div>
                        `;
                    }}

                    function bindPointClicks(chartElement) {{
                        chartElement.on('plotly_click', function(eventData) {{
                            const point = eventData.points[0];
                            renderDetails(point.pointIndex);
                        }});
                    }}

                    const traces = [
                        {{
                            x: rows.map((row) => row.date),
                            y: rows.map((row) => row.temp_max_c),
                            type: 'scatter',
                            mode: 'lines+markers',
                            name: 'Daily max',
                            line: {{ color: '#f24822', width: 2.2 }},
                            marker: {{ color: '#f24822', size: 6 }},
                            customdata: rows.map((row) => row),
                            hovertemplate: '%{{x}}<br>Max: %{{y}} °C<extra></extra>',
                        }},
                        {{
                            x: rows.map((row) => row.date),
                            y: rows.map((row) => row.temp_min_c),
                            type: 'scatter',
                            mode: 'lines+markers',
                            name: 'Daily min',
                            line: {{ color: '#228be6', width: 2.2 }},
                            marker: {{ color: '#228be6', size: 6 }},
                            customdata: rows.map((row) => row),
                            hovertemplate: '%{{x}}<br>Min: %{{y}} °C<extra></extra>',
                        }},
                    ];

                    const precipTraces = [
                        {{
                            x: rows.map((row) => row.date),
                            y: rows.map((row) => row.rain_mm),
                            type: 'scatter',
                            mode: 'lines+markers',
                            name: 'Rain',
                            line: {{ color: '#16a34a', width: 2.2 }},
                            marker: {{ color: '#16a34a', size: 6 }},
                            customdata: rows.map((row) => row),
                            hovertemplate: '%{{x}}<br>Rain: %{{y}} mm<extra></extra>',
                        }},
                        {{
                            x: rows.map((row) => row.date),
                            y: rows.map((row) => row.snowfall_cm),
                            type: 'scatter',
                            mode: 'lines+markers',
                            name: 'Snowfall',
                            line: {{ color: '#60a5fa', width: 2.2, dash: 'dot' }},
                            marker: {{ color: '#60a5fa', size: 6 }},
                            customdata: rows.map((row) => row),
                            hovertemplate: '%{{x}}<br>Snowfall: %{{y}} cm<extra></extra>',
                            yaxis: 'y2',
                        }},
                    ];

                    const layout = {{
                        title: {{ text: `${{cityName}} climate trend - previous 1 year` }},
                        margin: {{ l: 55, r: 20, t: 50, b: 45 }},
                        height: 420,
                        paper_bgcolor: 'rgba(0,0,0,0)',
                        plot_bgcolor: 'rgba(0,0,0,0)',
                        hovermode: 'x unified',
                        legend: {{ orientation: 'h', y: 1.12 }},
                        xaxis: {{ title: 'Date', showgrid: true, gridcolor: 'rgba(148,163,184,0.18)' }},
                        yaxis: {{ title: 'Temperature (°C)', showgrid: true, gridcolor: 'rgba(148,163,184,0.18)' }},
                    }};

                    const precipLayout = {{
                        title: {{ text: `${{cityName}} rain and snowfall - previous 1 year` }},
                        margin: {{ l: 55, r: 55, t: 50, b: 45 }},
                        height: 380,
                        paper_bgcolor: 'rgba(0,0,0,0)',
                        plot_bgcolor: 'rgba(0,0,0,0)',
                        hovermode: 'x unified',
                        legend: {{ orientation: 'h', y: 1.12 }},
                        xaxis: {{ title: 'Date', showgrid: true, gridcolor: 'rgba(148,163,184,0.18)' }},
                        yaxis: {{ title: 'Rain (mm)', showgrid: true, gridcolor: 'rgba(148,163,184,0.18)' }},
                        yaxis2: {{
                            title: 'Snowfall (cm)',
                            overlaying: 'y',
                            side: 'right',
                            showgrid: false,
                        }},
                    }};

                    Plotly.newPlot(chartEl, traces, layout, {{ responsive: true, displayModeBar: true, displaylogo: false }});
                    Plotly.newPlot(precipChartEl, precipTraces, precipLayout, {{ responsive: true, displayModeBar: true, displaylogo: false }});
                    bindPointClicks(chartEl);
                    bindPointClicks(precipChartEl);
                    renderDetails(rows.length - 1);
                }})();
            </script>
        </section>
        """


def build_dashboard_html(
    observations: pd.DataFrame,
    daily_df: pd.DataFrame,
    trend_frames: dict[str, pd.DataFrame],
    generated_at: str,
) -> str:
    latest = observations.sort_values("scraped_at_utc").groupby(
        "city", as_index=False).tail(1)
    latest = latest.sort_values("city")

    cards: list[str] = []
    for _, row in latest.iterrows():
        city_slug = slugify(str(row["city"]))
        cards.append(
            """
            <a class="card card-link" href="#{city_slug}-section">
              <h3>{city}</h3>
              <p class="temp">{temp} degC</p>
              <p>Humidity: {humidity}% | Wind: {wind} km/h</p>
              <p>Rain now: {rain} mm</p>
              <p class="meta">Last update: {updated}</p>
            </a>
            """.format(
                city_slug=city_slug,
                city=html.escape(str(row["city"])),
                temp=row.get("temp_c", "n/a"),
                humidity=row.get("humidity_pct", "n/a"),
                wind=row.get("wind_kmh", "n/a"),
                rain=row.get("precip_mm", "n/a"),
                updated=html.escape(str(row["scraped_at_utc"])),
            )
        )

    chart_blocks: list[str] = []
    for city, city_rows in sorted(trend_frames.items()):
        city_rows = city_rows.copy().sort_values("date")
        if city_rows.empty:
            continue
        window_start = city_rows["date"].min().strftime("%Y-%m-%d")
        window_end = city_rows["date"].max().strftime("%Y-%m-%d")
        chart_blocks.append(
            build_city_chart_section(city_rows, city)
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
        .card-link {{
            display: block;
            color: inherit;
            text-decoration: none;
            transition: transform 0.18s ease, box-shadow 0.18s ease, border-color 0.18s ease;
            cursor: pointer;
        }}
        .card-link:hover, .card-link:focus-visible {{
            transform: translateY(-2px);
            border-color: #93c5fd;
            box-shadow: 0 14px 34px rgba(15, 23, 42, 0.12);
            outline: none;
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
        .interactive-chart {{ width: 100%; min-height: 420px; }}
        .day-details {{
            margin-top: 0.75rem;
            padding: 0.85rem 0.95rem;
            border-radius: 12px;
            border: 1px solid #dbeafe;
            background: linear-gradient(180deg, rgba(248,250,252,0.96), rgba(239,246,255,0.96));
        }}
            .metric-panels {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
                gap: 0.75rem;
                margin-top: 0.75rem;
            }}
            .metric-panel {{
                border: 1px solid #dbeafe;
                border-radius: 12px;
                padding: 0.85rem 0.95rem;
                background: rgba(255, 255, 255, 0.84);
            }}
            .metric-panel span {{
                display: block;
                font-size: 0.78rem;
                color: var(--muted);
                text-transform: uppercase;
                letter-spacing: 0.04em;
                margin-bottom: 0.2rem;
            }}
            .metric-value {{
                display: block;
                font-size: 1.15rem;
                font-weight: 700;
                color: var(--ink);
            }}
            .metric-note {{
                margin: 0.25rem 0 0;
                color: var(--muted);
                font-size: 0.86rem;
            }}
        .detail-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
            gap: 0.7rem;
        }}
        .detail-grid span {{
            display: block;
            font-size: 0.78rem;
            color: var(--muted);
            text-transform: uppercase;
            letter-spacing: 0.04em;
            margin-bottom: 0.2rem;
        }}
        .detail-grid strong {{
            display: block;
            color: var(--ink);
            font-size: 1rem;
        }}
    @media (max-width: 700px) {{
      .container {{ width: min(1150px, calc(100% - 1rem)); }}
      .hero {{ padding: 0.9rem; }}
      h1 {{ font-size: 1.45rem; }}
    }}
  </style>
    <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
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
    trend_frames: dict[str, pd.DataFrame] = {}

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
        trend_frames[city.name] = trend_df

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
        all_observations, all_daily, trend_frames, run_ts)
    DASHBOARD_HTML.write_text(dashboard, encoding="utf-8")

    print(f"Wrote {len(all_daily_to_write)} daily weather rows to {DAILY_CSV}.")
    print(
        f"Wrote {len(all_observations)} observation rows to {OBSERVATIONS_CSV}.")
    print(f"Dashboard generated at {DASHBOARD_HTML}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
