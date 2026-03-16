import os
import logging
from fastapi import FastAPI, HTTPException, Request, Query
from pydantic import BaseModel, Field
from pymongo import MongoClient
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
from bson.objectid import ObjectId
from pydantic import GetJsonSchemaHandler
from pydantic.json_schema import JsonSchemaValue
from starlette.middleware.base import BaseHTTPMiddleware
import asyncio
from apscheduler.schedulers.asyncio import AsyncIOScheduler
import time

from analyzers.esm import ESMCalculator 
from pipeline_orchestrator import PipelineOrchestrator

os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/api.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("bse_api")


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


class PyObjectId(ObjectId):
    @classmethod
    def __get_pydantic_core_schema__(cls, _source_type, handler):
        return handler(str)

    @classmethod
    def __get_pydantic_json_schema__(
        cls, core_schema, handler: GetJsonSchemaHandler
    ) -> JsonSchemaValue:
        json_schema = handler(core_schema)
        json_schema.update(type="string")
        return json_schema


class MongoBaseModel(BaseModel):
    model_config = {
        "populate_by_name": True,
        "arbitrary_types_allowed": True,
        "json_encoders": {
            ObjectId: lambda v: str(v),
            datetime: lambda v: v.isoformat(),
        }
    }

class InsightResponse(MongoBaseModel):
    id: PyObjectId = Field(alias="_id")
    company: str
    date: str
    category: Optional[str] = None
    metrics: Dict[str, Any]


# Streamlined Ranking model for dashboard consumption
class RankingResponse(MongoBaseModel):
    id: PyObjectId = Field(alias="_id")
    company: str
    ticker: Optional[str] = None
    date: str
    ranking_score: Optional[float] = None
    
    # Flattened Metrics (aligned with esm_pipeline.py final output)
    quarter: Optional[str] = None
    sentiment: Optional[float] = None # Simplified key name from sentiment_score_finbert
    
    revenue_current: Optional[float] = None
    pat_current: Optional[float] = None
    ebitda_current: Optional[float] = None
    revenue_yoy_change: Optional[float] = None
    
    key_highlights: Optional[List[str]] = []


# Model for the forthcoming results calendar endpoint
class CalendarRecord(MongoBaseModel):
    scrip_code: str
    company_name: str
    short_name: str
    meeting_date_raw: str
    meeting_date_standard: Optional[str] = None
    bse_url: Optional[str] = None
    scraped_at: datetime


MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
DB_NAME = os.getenv("DB_NAME", "bse_data")

mongo_client = MongoClient(MONGO_URI)
db = mongo_client[DB_NAME]
esm_calculator: Optional[ESMCalculator] = None
orchestrator: Optional[PipelineOrchestrator] = None
scheduler: Optional[AsyncIOScheduler] = None

insights_collection = db["insights"]
parsed_pdfs_collection = db["parsed_pdfs"]
announcements_collection = db["announcements"]
result_calendar_collection = db["result_calendar"]

rankings_collection = db["rankings"] 

STARTUP_BACKFILL_DAYS = _env_int("STARTUP_BACKFILL_DAYS", 30)
STARTUP_LLM_LIMIT = _env_int("STARTUP_LLM_LIMIT", 250)
DAILY_LLM_LIMIT = _env_int("DAILY_LLM_LIMIT", 120)
DAILY_CRON_HOUR = _env_int("DAILY_CRON_HOUR", 18)
DAILY_CRON_MINUTE = _env_int("DAILY_CRON_MINUTE", 0)


class LogResponseMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        logger.info(f"Request: {request.method} {request.url.path}")
        response = await call_next(request)
        logger.info(f"Response Status: {response.status_code} ({request.method} {request.url.path})")
        return response


def safe_float(value, default=None):
    if value is None:
        return default
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def format_insight_for_ranking(doc: dict) -> dict:
    """Formats a MongoDB rankings document into the streamlined RankingResponse structure."""
    return {
        "_id": str(doc["_id"]),
        "company": doc.get("company"),
        "ticker": doc.get("ticker"),
        "date": doc.get("date"),
        "ranking_score": safe_float(doc.get("ranking_score")),
        "quarter": doc.get("quarter"),
        "sentiment": safe_float(doc.get("sentiment")),
        "revenue_current": safe_float(doc.get("revenue_current")),
        "pat_current": safe_float(doc.get("pat_current")),
        "ebitda_current": safe_float(doc.get("ebitda_current")),
        "revenue_yoy_change": safe_float(doc.get("revenue_yoy_change")),
        "key_highlights": doc.get("key_highlights", []),
    }


async def daily_pipeline_job():
    global orchestrator
    if not orchestrator:
        logger.error("Scheduler failed: Pipeline Orchestrator is not initialized.")
        return
    try:
        results = await orchestrator.run_daily_pipeline(limit=DAILY_LLM_LIMIT)
        logger.info("Daily pipeline completed: %s", results)
        return results
    except Exception as e:
        logger.error(f"Daily pipeline failed: {e}", exc_info=True)


def should_run_initial_backfill() -> bool:
    """Run long backfill only for a new/empty project state."""
    return announcements_collection.estimated_document_count() == 0


async def lifespan(app: FastAPI):
    global esm_calculator, orchestrator, scheduler

    try:
        esm_calculator = ESMCalculator()
        orchestrator = PipelineOrchestrator(MONGO_URI, DB_NAME)
        logger.info("ESM Calculator and Pipeline Orchestrator initialized")
    except Exception as e:
        logger.error(f"Failed to initialize ESM Calculator or Pipeline Orchestrator: {e}", exc_info=True)
        yield
        return

    try:
        scheduler = AsyncIOScheduler()

        scheduler.add_job(
            daily_pipeline_job,
            "cron",
            day_of_week="mon-fri",
            hour=DAILY_CRON_HOUR,
            minute=DAILY_CRON_MINUTE,
            id="daily_pipeline_job"
        )

        scheduler.start()
        logger.info(
            "APScheduler started with weekday daily cron at %02d:%02d.",
            DAILY_CRON_HOUR,
            DAILY_CRON_MINUTE,
        )
    except Exception as e:
        logger.error(f"Scheduler setup failed: {e}")


    try:
        if should_run_initial_backfill():
            logger.info(
                "Initial startup detected. Running %d-day backfill pipeline.",
                STARTUP_BACKFILL_DAYS,
            )
            result = await orchestrator.run_startup_backfill(
                days=STARTUP_BACKFILL_DAYS,
                limit=STARTUP_LLM_LIMIT,
            )
            logger.info("Startup backfill completed: %s", result)
        else:
            logger.info(
                "Startup backfill skipped (announcements already present). "
                "Daily cron will continue incremental processing."
            )
    except Exception as e:
        logger.error(f"Startup pipeline failed: {e}", exc_info=True)

    yield

    if scheduler:
        scheduler.shutdown()
        logger.info("APScheduler shut down.")
    if mongo_client:
        mongo_client.close()
        logger.info("MongoDB connection closed.")


app = FastAPI(
    title="BSE Insights API",
    lifespan=lifespan
)
app.add_middleware(LogResponseMiddleware)


@app.get("/insights", response_model=List[InsightResponse])
def get_insights(limit: int = 20):
    docs = insights_collection.find().sort("analyzed_at", -1).limit(limit)
    return list(docs)


@app.get("/rankings", response_model=Dict[str, List[RankingResponse]])
def get_rankings(
    start_date: str = Query((datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")),
    end_date: str = Query(datetime.now().strftime("%Y-%m-%d"))
):
    query = {
        "date": {"$gte": start_date, "$lte": end_date},
        "ranking_score": {"$exists": True, "$ne": None}
    }
    # MODIFIED: Querying the new rankings_collection instead of insights_collection
    ranked_docs = list(rankings_collection.find(query).sort("ranking_score", -1))
    if not ranked_docs:
        return {"rankings": []}
        
    rankings = [format_insight_for_ranking(doc) for doc in ranked_docs]
    return {"rankings": rankings}


@app.get("/rankings/summary")
def get_ranking_summary(
    start_date: str = Query((datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")),
    end_date: str = Query(datetime.now().strftime("%Y-%m-%d"))
):
    query = {
        "date": {"$gte": start_date, "$lte": end_date},
        "ranking_score": {"$exists": True, "$ne": None}
    }
    pipeline = [
        {"$match": query},
        {"$group": {
            "_id": None,
            "total_ranked": {"$sum": 1},
            "avg_ranking_score": {"$avg": "$ranking_score"},
            "max_ranking_score": {"$max": "$ranking_score"},
            "min_ranking_score": {"$min": "$ranking_score"},
            # NOTE: We assume 'ranking_score' being > 0 is the positive signal proxy.
            "total_positive_signals": {"$sum": {"$cond": [{"$gt": ["$ranking_score", 0]}, 1, 0]}}
        }}
    ]
    # MODIFIED: Aggregating on the new rankings_collection
    summary = list(rankings_collection.aggregate(pipeline))
    if not summary:
        return {
            "total_ranked": 0,
            "avg_ranking_score": 0.0,
            "max_ranking_score": 0.0,
            "min_ranking_score": 0.0,
            "total_positive_signals": 0
        }
    return summary[0]

@app.get("/calendar")
def get_forthcoming_results():
    """Retrieves forthcoming results (today or future) from the calendar collection."""
    today_str = datetime.now().strftime("%Y-%m-%d")
    
    query = {
        "meeting_date_standard": {"$gte": today_str},
        "short_name": {"$ne": None} # Ensure we only return records with a ticker
    }
    
    # Fetch and sort by meeting date
    docs = list(result_calendar_collection.find(query).sort("meeting_date_standard", 1))
    
    # Manual data preparation is minimal due to CalendarRecord model alignment
    return {"calendar": docs}


@app.post("/pipeline/run")
async def run_full_pipeline_endpoint(
    days_ago: int = Query(STARTUP_BACKFILL_DAYS),
    limit: int = Query(STARTUP_LLM_LIMIT)
):
    if not orchestrator:
        raise HTTPException(503, "Pipeline orchestrator not initialized.")
    try:
        results = await orchestrator.run_full_pipeline(days_ago=days_ago, limit=limit)
        return {"status": "Pipeline run complete", "details": results}
    except Exception as e:
        logger.error(f"Error running pipeline: {e}", exc_info=True)
        raise HTTPException(500, f"Pipeline execution failed: {str(e)}")

@app.get("/pipeline/stats")
def get_pipeline_stats():
    return {
        "announcements_count": announcements_collection.count_documents({}),
        "parsed_pdfs_count": parsed_pdfs_collection.count_documents({}),
        "insights_count": insights_collection.count_documents({}),
        "insights_with_ranking_score": insights_collection.count_documents({"ranking_score": {"$exists": True, "$ne": None}}),
        "calendar_count": result_calendar_collection.count_documents({}),
        "rankings_count": rankings_collection.count_documents({})
    }
