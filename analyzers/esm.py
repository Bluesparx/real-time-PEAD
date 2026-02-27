#!/usr/bin/env python3
"""
esm_pipeline.py

Full clean rewrite of your ESM pipeline (FULL CLEAN VERSION).
- Single Gemini call for batch ranking of insights.
- Keeps DB document structure unchanged (history under price docs,
  insights documents preserved with `_id`).
- Compact imports, removed unused helpers and unused metrics.
"""

import os
import time
import json
import math
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional

import pandas as pd
import numpy as np
import yfinance as yf
from pymongo import MongoClient, UpdateOne
from pymongo.errors import BulkWriteError
from bson.objectid import ObjectId

# NOTE: google.genai usage retained; require GOOGLE GENAI client installed & configured
from google import genai

# ---------------------------
# Configuration & Logging
# ---------------------------
LOG_DIR = os.getenv("LOG_DIR", "logs")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_DATE = datetime.now().strftime("%Y-%m-%d")
LOG_FILE = os.path.join(LOG_DIR, f"esm_pipeline_{LOG_DATE}.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(LOG_FILE, mode="a"), logging.StreamHandler()],
)
logger = logging.getLogger("ESM_Pipeline")

# Mongo config
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
DB_NAME = os.getenv("DB_NAME", "bse_data")
PRICE_COLLECTION = os.getenv("PRICE_COLLECTION", "stock_prices")
INSIGHTS_COLLECTION = os.getenv("INSIGHTS_COLLECTION", "insights")
RANKINGS_COLLECTION = os.getenv("RANKINGS_COLLECTION", "rankings")

# Other defaults
TICKER_CSV_PATH = os.getenv("TICKER_CSV_PATH", "companies.csv")
MARKET_TICKER = os.getenv("MARKET_TICKER", "BSE-200.BO")
ESTIMATION_WINDOW = int(os.getenv("ESTIMATION_WINDOW", "200"))
EVENT_WINDOW_DAYS = int(os.getenv("EVENT_WINDOW_DAYS", "7"))

# Gemini / GenAI client
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyA5dlO-h-uv5XBEuRpzBXj1l8qOzp9oyow")
if not GEMINI_API_KEY:
    logger.warning("GEMINI_API_KEY not set in environment. Set GEMINI_API_KEY before running.")
gemini_client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None


# ---------------------------
# Utilities
# ---------------------------
def get_mongo_client() -> MongoClient:
    client = MongoClient(MONGO_URI)
    try:
        client.admin.command("ping")
    except Exception as e:
        logger.error(f"MongoDB ping failed: {e}")
        raise
    return client


def date_to_string(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d")


# ---------------------------
# Stock Price Manager
# ---------------------------
class StockPriceManager:
    def __init__(self, mongo_uri: str = MONGO_URI, db_name: str = DB_NAME):
        self.client = MongoClient(mongo_uri)
        self.db = self.client[db_name]
        self.col = self.db[PRICE_COLLECTION]

    @staticmethod
    def _normalize_ticker(ticker: str) -> str:
        return ticker.strip()

    def _fetch_prices_yfinance(self, yahoo_ticker: str, start_date: datetime, end_date: datetime) -> List[Dict[str, Any]]:
        """Download via yfinance and transform to list of {date, close, volume} dicts."""
        try:
            df = yf.download(yahoo_ticker, start=start_date, end=end_date + timedelta(days=1), progress=False)
        except Exception as e:
            logger.error(f"yfinance download failed for {yahoo_ticker}: {e}")
            return []

        if df.empty:
            return []

        # Flatten columns if MultiIndex
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = df.reset_index()
        rows = []
        for _, r in df.iterrows():
            date_val = r.get("Date")
            date_str = (pd.to_datetime(date_val)).strftime("%Y-%m-%d") if date_val is not None else None
            close_val = r.get("Close", None)
            volume_val = r.get("Volume", None)
            if pd.isna(close_val):
                close = None
            else:
                close = float(close_val)
            if pd.isna(volume_val):
                volume = None
            else:
                volume = int(volume_val)
            rows.append({"date": date_str, "close": close, "volume": volume})
        return rows

    def backfill_history(self, ticker: str, start_date: Optional[str] = None, end_date: Optional[str] = None):
        if isinstance(start_date, str):
            start_date_dt = datetime.strptime(start_date, "%Y-%m-%d").date()
        else:
            start_date_dt = None
        if isinstance(end_date, str):
            end_date_dt = datetime.strptime(end_date, "%Y-%m-%d").date()
        else:
            end_date_dt = datetime.now().date()

        end_date_dt = end_date_dt or datetime.now().date()
        start_date_dt = start_date_dt or (end_date_dt - timedelta(days=365))

        yahoo_ticker = self._normalize_ticker(ticker)
        fetched = self._fetch_prices_yfinance(yahoo_ticker, start_date_dt, end_date_dt)
        if not fetched:
            logger.info(f"No fetched rows for {yahoo_ticker}")
            return

        doc = self.col.find_one({"ticker": yahoo_ticker}, {"history.date": 1})
        existing_dates = set()
        if doc and "history" in doc:
            existing_dates = {h["date"] for h in doc["history"] if h.get("date")}

        new_rows = [r for r in fetched if r.get("date") not in existing_dates]
        if new_rows:
            self.col.update_one(
                {"ticker": yahoo_ticker},
                {"$push": {"history": {"$each": new_rows}}, "$set": {"base_ticker": ticker if ticker != MARKET_TICKER else None}},
                upsert=True,
            )
            logger.info(f"Pushed {len(new_rows)} new history rows for {yahoo_ticker}")
        else:
            logger.info(f"No new history rows for {yahoo_ticker}")

    def update_daily_price(self, ticker: str):
        yahoo_ticker = self._normalize_ticker(ticker)
        today = datetime.now().date().strftime("%Y-%m-%d")
        doc = self.col.find_one({"ticker": yahoo_ticker}, {"history.date": 1})
        if doc and "history" in doc and any(h.get("date") == today for h in doc["history"]):
            logger.debug(f"{yahoo_ticker} already has today's price.")
            return

        prices = self._fetch_prices_yfinance(yahoo_ticker, datetime.now().date() - timedelta(days=1), datetime.now().date())
        if not prices:
            logger.info(f"No price fetched today for {yahoo_ticker}")
            return
        self.col.update_one(
            {"ticker": yahoo_ticker},
            {"$push": {"history": {"$each": prices}}, "$set": {"base_ticker": ticker if ticker != MARKET_TICKER else None}},
            upsert=True,
        )
        logger.info(f"Updated daily prices for {yahoo_ticker}")

    def get_price_history(self, ticker: str) -> List[Dict[str, Any]]:
        doc = self.col.find_one({"ticker": ticker})
        if not doc:
            return []
        return doc.get("history", [])


# ---------------------------
# ESM Calculator (Batch LLM Ranking)
# ---------------------------
class ESMCalculator:
    def __init__(self, mongo_uri: str = MONGO_URI, db_name: str = DB_NAME):
        self.client = MongoClient(mongo_uri)
        self.db = self.client[db_name]
        self.price_col = self.db[PRICE_COLLECTION]
        self.insights_col = self.db[INSIGHTS_COLLECTION]
        self.rankings_col = self.db[RANKINGS_COLLECTION]
        self._price_cache: Dict[str, pd.DataFrame] = {}
        self.market_ticker = MARKET_TICKER

    # ---------------------------
    # Price helpers
    # ---------------------------
    def _fetch_price_df(self, ticker: str) -> Optional[pd.DataFrame]:
        """Load price history from Mongo and cache as DataFrame indexed by datetime."""
        if not ticker:
            return None
        if ticker in self._price_cache:
            return self._price_cache[ticker]

        doc = self.price_col.find_one({"ticker": ticker}, {"history": 1})
        if not doc or not doc.get("history"):
            return None

        df = pd.DataFrame(doc["history"])
        if df.empty:
            return None

        if "date" not in df.columns:
            return None
        df = df.set_index("date")
        # Ensure datetime index
        df.index = pd.to_datetime(df.index)
        if "close" in df.columns:
            df["close"] = pd.to_numeric(df["close"], errors="coerce")
        self._price_cache[ticker] = df.sort_index()
        return self._price_cache[ticker]

    def _compute_event_price_stats(self, ticker: str, event_dt: datetime) -> Dict[str, Any]:
        """Compute estimation stats and event window returns for use in LLM prompt."""
        result = {"available": False, "estimation_stats": {}, "event_window_returns": [], "event_days": 0}
        stock_df = self._fetch_price_df(ticker)
        market_df = self._fetch_price_df(self.market_ticker)
        if stock_df is None or market_df is None:
            return result

        # compute daily returns
        if "close" not in stock_df.columns or "close" not in market_df.columns:
            return result

        stock_df = stock_df.copy()
        market_df = market_df.copy()
        stock_df["Stock_Return"] = stock_df["close"].pct_change()
        market_df["Market_Return"] = market_df["close"].pct_change()

        merged = pd.merge(stock_df[["Stock_Return"]], market_df[["Market_Return"]], left_index=True, right_index=True).dropna()
        if merged.empty:
            return result

        event_dt = pd.to_datetime(event_dt)
        estimation_end = merged.index[merged.index < event_dt]
        if estimation_end.empty:
            return result
        estimation_end = estimation_end.max()
        estimation_df = merged.loc[:estimation_end].tail(ESTIMATION_WINDOW)

        if estimation_df.empty or len(estimation_df) < 50:
            # not enough estimation data
            return result

        # event window centered on event date
        half_win = EVENT_WINDOW_DAYS // 2
        event_start = event_dt - pd.Timedelta(days=half_win)
        event_end = event_dt + pd.Timedelta(days=half_win)
        event_window_df = merged.loc[event_start:event_end]

        result["available"] = not event_window_df.empty
        result["estimation_stats"] = {
            "stock_mean_return": float(estimation_df["Stock_Return"].mean()),
            "market_mean_return": float(estimation_df["Market_Return"].mean()),
            "correlation": float(estimation_df["Stock_Return"].corr(estimation_df["Market_Return"])),
            "estimation_samples": int(len(estimation_df)),
        }
        if not event_window_df.empty:
            result["event_window_returns"] = (
                event_window_df[["Stock_Return", "Market_Return"]].reset_index().to_dict(orient="records")
            )
            result["event_days"] = len(event_window_df)
        else:
            result["event_window_returns"] = []
            result["event_days"] = 0
        return result

    # ---------------------------
    # Single Gemini batch call
    # ---------------------------
    def _get_batch_llm_rankings(self, insights_batch: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Calls Gemini ONCE with all insights and returns mapping: { insight_id_str: ranking_score }.
        The prompt provides fundamentals, sentiment and price snapshot (if available).
        """
        if gemini_client is None:
            logger.error("Gemini client not initialized - GEMINI_API_KEY missing.")
            return {}

        compact = []
        for ins in insights_batch:
            ins_id = str(ins["_id"])
            company = ins.get("company")
            event_date = ins.get("date")
            metrics = ins.get("metrics", {})
            sentiment = ins.get("sentiment_score", 0)

            # Try to compute price snapshot for this insight's company
            ticker_guess = None
            # if insight already has esm_ticker, use it; otherwise attempt to fallback to metrics.ticker if present
            if ins.get("esm_ticker"):
                ticker_guess = ins.get("esm_ticker")
            elif metrics.get("ticker"):
                ticker_guess = metrics.get("ticker")
            else:
                # No guaranteed ticker: leave None
                ticker_guess = None

            price_snapshot = {}
            if ticker_guess:
                try:
                    price_snapshot = self._compute_event_price_stats(ticker_guess, event_date)
                except Exception as e:
                    logger.debug(f"Price snapshot compute failed for {ticker_guess}@{event_date}: {e}")
                    price_snapshot = {}

            compact.append(
                {
                    "id": ins_id,
                    "company": company,
                    "event_date": event_date,
                    "fundamentals": metrics,
                    "sentiment_score": sentiment,
                    "price_snapshot": price_snapshot,
                }
            )

        # Build the LLM prompt
        # Keep the instructions strict: JSON array of {id, ranking_score}
        prompt = (
            "You are an expert financial analyst. You will be given a JSON array of earnings events.\n"
            "For each event, produce a single 'ranking_score' between -1.0 (worst) and 1.0 (best).\n"
            "Consider: fundamentals (PAT, EBITDA, revenue growth), sentiment_score, price snapshots (if available), and contextual data (key highlights).\n"
            "Weight allocation: fundamentals (40%), sentiment_score (25%), price snapshot (20%), contextual data (15%).\n"
            "Return strictly a JSON array where each element is `{ \"id\": \"<id>\", \"ranking_score\": <float> }`."
            "Do NOT include any other text.\n\n"
            "INPUT:\n"
            f"{json.dumps(compact, indent=2)}\n"
        )

        # call gemini once
        try:
            response = gemini_client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
            # small pause if necessary
            time.sleep(0.5)
            text = response.text.strip()
            # remove triple backticks if present
            text = text.replace("```json", "").replace("```", "").strip()
            parsed = json.loads(text)
            if not isinstance(parsed, list):
                logger.error("Unexpected Gemini response format: expected JSON array.")
                return {}
            mapping = {}
            for elem in parsed:
                _id = str(elem.get("id"))
                score = float(elem.get("ranking_score", 0.0))
                mapping[_id] = score
            return mapping
        except Exception as e:
            logger.error(f"Gemini batch ranking call failed: {e}")
            return {}

    # ---------------------------
    # Run bulk analysis (entrypoint)
    # ---------------------------
    def run_bulk_analysis(self, start_date_str: str, end_date_str: str) -> List[Dict[str, Any]]:
        """
        High-level: fetch unranked insights in date range, compute a single batch Gemini call
        to obtain ranking_scores for all, then update insights collection and upsert into rankings.
        Returns list of final ranking docs (sorted).
        """
        # Defensive date parsing
        start_date_cutoff = start_date_str
        end_date_cutoff = end_date_str

        query = {
            "date": {"$gte": start_date_cutoff, "$lte": end_date_cutoff},
            "ranking_score": {"$exists": False},
        }
        logger.info(f"Querying insights: {query}")
        insights_cursor = self.insights_col.find(query).sort("date", -1)
        insights_to_process = list(insights_cursor)
        logger.info(f"Found {len(insights_to_process)} unranked insights.")

        if not insights_to_process:
            return []

        # single LLM call for all insights
        logger.info("Preparing batch LLM payload for Gemini...")
        ranking_map = self._get_batch_llm_rankings(insights_to_process)
        logger.info(f"Received {len(ranking_map)} ranking entries from Gemini.")

        # Prepare bulk DB ops
        insights_update_ops = []
        rankings_upsert_ops = []
        results = []

        for ins in insights_to_process:
            ins_id = ins["_id"]
            ins_id_str = str(ins_id)
            company = ins.get("company")
            metrics = ins.get("metrics", {})
            sentiment = ins.get("sentiment_score", 0)

            score = float(ranking_map.get(ins_id_str, 0.0))
            # determine esm_ticker (best-effort): prefer existing field, otherwise attempt using metrics.ticker
            esm_ticker = ins.get("esm_ticker") or metrics.get("ticker") or None

            # update original insight doc with ranking_score and esm_ticker
            insights_update_ops.append(
                UpdateOne(
                    {"_id": ins_id},
                    {
                        "$set": {"ranking_score": float(f"{score:.6f}"), "esm_ticker": esm_ticker},
                        "$unset": {"esm_car": "", "composite_score": "", "car_period_days": ""},
                    },
                )
            )

            # final flattened doc for rankings collection (preserve _id)
            final_doc = {
                "_id": ins_id,
                "company": company,
                "ticker": esm_ticker,
                "date": ins.get("date"),
                "ranking_score": float(f"{score:.6f}"),
                "quarter": metrics.get("quarter"),
                "sentiment": sentiment,
                "revenue_current": metrics.get("revenue_current_qtr"),
                "pat_current": metrics.get("pat_current_qtr"),
                "ebitda_current": metrics.get("ebitda_current_qtr"),
                "revenue_yoy_change": metrics.get("revenue_yoy_change_pct"),
                "key_highlights": metrics.get("key_highlights", []),
            }

            rankings_upsert_ops.append(UpdateOne({"_id": ins_id}, {"$set": final_doc}, upsert=True))
            results.append(final_doc)

        # execute bulk writes
        try:
            if insights_update_ops:
                res = self.insights_col.bulk_write(insights_update_ops, ordered=False)
                logger.info(f"Updated {res.modified_count} insight documents with ranking_score.")
        except BulkWriteError as bwe:
            logger.error(f"Bulk write error updating insights: {bwe.details}")

        try:
            if rankings_upsert_ops:
                res2 = self.rankings_col.bulk_write(rankings_upsert_ops, ordered=False)
                logger.info(f"Inserted/Updated {len(rankings_upsert_ops)} ranking documents (upsert).")
        except BulkWriteError as bwe:
            logger.error(f"Bulk write error upserting rankings: {bwe.details}")

        # sort and attach rank index
        results.sort(key=lambda x: x["ranking_score"], reverse=True)
        for idx, r in enumerate(results):
            r["rank"] = idx + 1

        # Optionally: produce a short rationale using Gemini for the top results (single extra call)
        # Keep this optional and lightweight. If you want it enabled, set environment var GENERATE_COMPARISON_RATIONALE=true
        if os.getenv("GENERATE_COMPARISON_RATIONALE", "false").lower() in ("1", "true", "yes"):
            try:
                top_for_rationale = results[:10]
                comp_prompt = (
                    "You are an expert financial analyst. Provide a concise (max 3-sentence) summary "
                    "explaining why the top-ranked company achieved the highest score compared to peers. "
                    "Focus on revenue YoY change and key highlights.\n\n"
                    f"DATA: {json.dumps(top_for_rationale, default=str, indent=2)}\n\n"
                    "Return only plain text summary (no JSON)."
                )
                resp = gemini_client.models.generate_content(model=GEMINI_MODEL, contents=comp_prompt)
                time.sleep(0.3)
                rationale = resp.text.strip()
                logger.info(f"Comparative Rationale:\n{rationale}")
            except Exception as e:
                logger.error(f"Failed to generate comparative rationale: {e}")

        logger.info(f"Bulk composite ranking analysis complete. Processed {len(results)} events.")
        return results


# ---------------------------
# External wrapper functions (pipeline entrypoints)
# ---------------------------
def run_esm_backfill(ticker_list: List[str], start_date: Optional[str] = None, end_date: Optional[str] = None):
    """
    Backfill historical prices for given tickers using StockPriceManager.
    ticker_list: list of tickers (strings)
    """
    spm = StockPriceManager()
    for t in ticker_list:
        try:
            spm.backfill_history(t, start_date=start_date, end_date=end_date)
        except Exception as e:
            logger.error(f"Backfill failed for {t}: {e}")
    spm.client.close()


def run_esm_daily_update(ticker_list: List[str]):
    spm = StockPriceManager()
    for t in ticker_list:
        try:
            spm.update_daily_price(t)
        except Exception as e:
            logger.error(f"Daily update failed for {t}: {e}")
    spm.client.close()


def run_esm_bulk_analysis(start_date: str, end_date: str):
    esm = ESMCalculator()
    logger.info(f"Running BULK COMPOSITE RANKING ANALYSIS for insights from {start_date} to {end_date}")
    results = esm.run_bulk_analysis(start_date, end_date)
    esm.client.close()
    return results

def esm_already_exists_for_today():
    """Check if ranking analysis has been done for today - checks for ranking_score presence"""
    try:
        client = get_mongo_client()
        db = client[DB_NAME]
        existing = db[INSIGHTS_COLLECTION].find_one({
            "date": datetime.now().strftime("%Y-%m-%d"),
            "ranking_score": {"$exists": True}
        })
        client.close()
        return existing is not None
    except Exception as e:
        logger.error(f"Error checking if Ranking exists for today: {e}")
        return False

# ---------------------------
# CLI
# ---------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ESM Pipeline (clean single-file)")
    parser.add_argument("mode", choices=["backfill", "daily_price_update", "bulk_esm_analysis"], help="Mode")
    parser.add_argument("--start", type=str, help="Start date YYYY-MM-DD (for backfill/analysis)", default=None)
    parser.add_argument("--end", type=str, help="End date YYYY-MM-DD (for backfill/analysis)", default=None)
    parser.add_argument("--tickers", type=str, help="Comma-separated tickers for backfill/daily", default=None)

    args = parser.parse_args()

    if args.mode == "backfill":
        tickers = args.tickers.split(",") if args.tickers else []
        run_esm_backfill(tickers, start_date=args.start, end_date=args.end)
    elif args.mode == "daily_price_update":
        tickers = args.tickers.split(",") if args.tickers else []
        run_esm_daily_update(tickers)
    elif args.mode == "bulk_esm_analysis":
        end_date = args.end if args.end else date_to_string(datetime.now())
        start_date = args.start if args.start else date_to_string(datetime.now() - timedelta(days=30))
        run_esm_bulk_analysis(start_date, end_date)