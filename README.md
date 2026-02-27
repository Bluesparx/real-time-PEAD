# BSE Earnings Insights Pipeline

Production-leaning local pipeline for:
- Scraping BSE result announcements
- Parsing attached PDFs and extracting structured metrics
- Running sentiment + ranking analysis
- Serving data via FastAPI
- Visualizing results in Streamlit

## Architecture

1. `scraper_cron.py`
- Pulls result announcements from BSE API.
- Upserts into MongoDB `announcements`.

2. `analyzers/llm_analyzer.py`
- Downloads announcement PDFs.
- Extracts text (native + OCR fallback).
- Uses Gemini for metrics extraction.
- Uses FinBERT for sentiment.
- Stores output in `parsed_pdfs` and `insights`.

3. `analyzers/esm.py`
- Builds ranking score using fundamentals, sentiment, and optional price context.
- Writes final flattened ranking docs into `rankings`.

4. `scrapers/result_calendar.py`
- Scrapes forthcoming results (next 30 days).
- Stores in `result_calendar`.

5. `pipeline_orchestrator.py`
- Orchestrates end-to-end flow:
  - announcements scraper
  - LLM pipeline
  - forthcoming calendar scraper
  - ESM ranking

6. `app.py`
- FastAPI server.
- Startup hooks, cron jobs, health endpoints, and pipeline endpoints.

7. `dashboard.py`
- Streamlit UI for rankings and result calendar.

## Local Setup

### 1. Create virtual environment and install dependencies

```powershell
cd C:\Users\nazia\Desktop\final-project
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
playwright install chromium
```

### 2. Configure environment

```powershell
copy .env.example .env
```

Update `.env` with your actual keys, especially `GEMINI_API_KEY`.

## Run Locally

Use 3 terminals.

### Terminal 1: MongoDB

```powershell
mongod
```

If MongoDB is already running as a Windows service, skip this.

### Terminal 2: FastAPI

```powershell
.\.venv\Scripts\Activate.ps1
python main.py
```

### Terminal 3: Streamlit dashboard

```powershell
cd C:\Users\nazia\Desktop\final-project
.\.venv\Scripts\Activate.ps1
streamlit run dashboard.py
```

Dashboard URL:
- `http://localhost:8501`

## Startup Jobs

Run announcement scraper for today:

```powershell
python scraper_cron.py today
```

Run forthcoming result calendar scraper:

```powershell
python scrapers\result_calendar.py
```

Trigger full pipeline from API:

```powershell
curl -X POST "http://localhost:8000/pipeline/run?days_ago=7&limit=50"
```

## Production Notes

- Keep secrets only in env vars, never in source.
- Consider moving scheduler out of API process for larger workloads.
- Add CI checks (`ruff`, `black`, tests) before deployments.
- Use a process manager (`systemd`, `supervisor`, Docker/Kubernetes) for reliability.
