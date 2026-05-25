# autotracker-ai

A safe MVP that collects car-related reference data from the public NHTSA vPIC API and stores updates in Excel.

## What this project does

- Fetches car makes and models from NHTSA public endpoints.
- Normalizes rows into a stable schema suitable for tracking.
- Writes output to:
  - `output/cars.xlsx`
  - `output/cars.csv`
- Runs daily via GitHub Actions and commits updated output files.

## Why some fields are empty

This MVP uses a public/government API for legal and reliability safety.
Marketplace-specific fields are included as optional columns and may be empty:

- `price`
- `location`
- `mileage`
- `transmission`
- `fuel`

## Local run

```bash
python -m venv .venv
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python scraper.py
```

## Environment variables

Optional runtime tuning:

- `MAX_MAKES` (default: `15`)
- `YEAR_COUNT` (default: `3`)

Example:

```bash
MAX_MAKES=10 YEAR_COUNT=2 python scraper.py
```

## GitHub Actions schedule

Workflow file: `.github/workflows/scrape.yml`

Default cron:

- `0 2 * * *` (daily at 02:00 UTC)

You can also run manually from the Actions tab using `workflow_dispatch`.

## File outputs

- Main Excel file: `output/cars.xlsx`
- Diff-friendly mirror: `output/cars.csv`

Yes, updates are stored in the Excel file and committed back to your repository when data changes.
