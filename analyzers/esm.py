import os
import time
import httpx
import pandas as pd
import numpy as np
import statsmodels.api as sm
from google import genai
import yfinance as yf
from pymongo import MongoClient, UpdateOne
from pymongo.errors import BulkWriteError
from bson.objectid import ObjectId
from datetime import datetime, timedelta
import logging
import math
from typing import List, Dict, Any, Optional, Tuple
import argparse 
import json

# Logging setup
log_date = datetime.now().strftime("%Y-%m-%d")
log_file = f"logs/esm_pipeline_{log_date}.log"
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('ESM_Pipeline')

# Mongo / DB setup
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
DB_NAME = os.getenv("DB_NAME", "bse_data")
PRICE_COLLECTION = "stock_prices" 
INSIGHTS_COLLECTION = "insights" 
TICKER_CSV_PATH = "companies.csv"

MARKET_TICKER = 'BSE-200.BO' 
ESTIMATION_WINDOW = 200     
EVENT_WINDOW_DAYS = 7

def get_mongo_client():
    try:
        client = MongoClient(MONGO_URI)
        client.admin.command('ping')
        return client
    except Exception as e:
        logger.error(f"Failed to connect to MongoDB: {e}")
        raise

def date_to_string(dt: datetime) -> str:
    return dt.strftime('%Y-%m-%d')


class StockPriceManager:
    def __init__(self, mongo_uri="mongodb://localhost:27017/", db_name="bse_data"):
        self.client = MongoClient(mongo_uri)
        self.db = self.client[db_name]
        self.col = self.db["stock_prices"]

    def _normalize_ticker(self, ticker):
        return f"{ticker}"

    def _fetch_prices(self, yahoo_ticker, start_date, end_date):
        df = yf.download(yahoo_ticker, start=start_date, end=end_date, progress=False)
        if df.empty:
            return []
        
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        df = df.reset_index()
        rows = []
        for _, r in df.iterrows():
            date_val = r["Date"]
            if isinstance(date_val, pd.Timestamp):
                date_str = date_val.strftime("%Y-%m-%d")
            elif hasattr(date_val, 'strftime'):
                date_str = date_val.strftime("%Y-%m-%d")
            else:
                date_str = str(date_val)[:10]
            
            close_val = r["Close"]
            volume_val = r["Volume"]
            
            if isinstance(close_val, pd.Series):
                close_val = close_val.iloc[0] if len(close_val) > 0 else None
            if isinstance(volume_val, pd.Series):
                volume_val = volume_val.iloc[0] if len(volume_val) > 0 else None
                
            rows.append({
                "date": date_str,
                "close": float(close_val) if close_val is not None and not pd.isna(close_val) else None,
                "volume": int(volume_val) if volume_val is not None and not pd.isna(volume_val) else None
            })
        return rows
    
    def _get_all_tickers(self) -> List[str]:
        tickers = self.col.distinct("ticker")
        return tickers

    def backfill_history(self, ticker, start_date=None, end_date=None):
        if isinstance(start_date, str):
            start_date = datetime.strptime(start_date, "%Y-%m-%d").date()
        if isinstance(end_date, str):
            end_date = datetime.strptime(end_date, "%Y-%m-%d").date()
        end_date = end_date or datetime.now().date()
        start_date = start_date or (end_date - timedelta(days=365))

        yahoo_ticker = self._normalize_ticker(ticker)
        fetched = self._fetch_prices(yahoo_ticker, start_date, end_date)
        if not fetched:
            return

        doc = self.col.find_one({"ticker": yahoo_ticker}, {"history.date": 1})
        existing_dates = set()
        if doc and "history" in doc:
            existing_dates = {h["date"] for h in doc["history"]}

        new_rows = [row for row in fetched if row["date"] not in existing_dates]

        if new_rows:
            self.col.update_one(
                {"ticker": yahoo_ticker},
                {
                    "$push": {"history": {"$each": new_rows}},
                    "$set": {"base_ticker": ticker if ticker != MARKET_TICKER else None}
                },
                upsert=True
            )

    def update_daily_price(self, ticker):
        yahoo_ticker = self._normalize_ticker(ticker)
        today = datetime.now().date().strftime("%Y-%m-%d")

        doc = self.col.find_one(
            {"ticker": yahoo_ticker},
            {"history.date": 1}
        )

        if doc and "history" in doc:
            if any(h["date"] == today for h in doc["history"]):
                return

        prices = self._fetch_prices(yahoo_ticker, datetime.now().date(), datetime.now().date() + timedelta(days=1))
        if not prices:
            return

        self.col.update_one(
            {"ticker": yahoo_ticker},
            {
                "$push": {"history": {"$each": prices}},
                "$set": {"base_ticker": ticker if ticker != MARKET_TICKER else None}
            },
            upsert=True
        )

    def get_price_history(self, ticker):
        yahoo_ticker = self._normalize_ticker(ticker)
        doc = self.col.find_one({"ticker": yahoo_ticker})
        if not doc:
            return []
        return doc.get("history", [])


class ESMCalculator:

    def __init__(self):
        self.client = get_mongo_client()
        self.db = self.client[DB_NAME]
        self.price_col = self.db[PRICE_COLLECTION]
        self.insights_col = self.db[INSIGHTS_COLLECTION]
        self.market_col = self.db[PRICE_COLLECTION] 
        self.ticker_map = self._load_ticker_map()
        self.GEMINI_MODEL = "gemini-2.5-flash" 
        self.gemini_client = genai.Client(api_key="AIzaSyBSUGneVbZruEhpXWkaOPwIzv_QjOzjMWs")
        self._price_cache: Dict[str, pd.DataFrame] = {}

    def _clean_company_name(self, company_name: str) -> str:
        return company_name.upper().replace('LIMITED', '').replace('LTD.', '').replace('LTD', '').replace('.', '').strip()

    def _load_ticker_map(self) -> Dict[str, str]:
        if not os.path.exists(TICKER_CSV_PATH):
            logger.error(f"TICKER CSV not found at {TICKER_CSV_PATH}")
            return {}

        try:
            df = pd.read_csv(TICKER_CSV_PATH)
            
            name_col = next((col for col in df.columns if 'company' in col.lower() or 'name' in col.lower()), None)
            ticker_col = "TICKER"

            if not name_col or ticker_col not in df.columns:
                logger.error("Could not find required Company Name or TICKER columns in CSV")
                return {}
            
            df['clean_name'] = df[name_col].astype(str).str.upper().str.replace(r'LTD\.?', '', regex=True).str.replace(r'\s+LTD\s*', '', regex=True).str.replace(r'\.', '', regex=False).str.strip()
            df['ticker'] = df[ticker_col].astype(str)
            
            ticker_map = df.set_index('clean_name')['ticker'].to_dict()
            
            logger.info(f"Loaded {len(ticker_map)} ticker mappings into memory")
            return ticker_map
        except Exception as e:
            logger.error(f"Error loading ticker map from CSV: {e}")
            return {}

    def _get_ticker_by_company_name(self, company_name: str) -> Optional[str]:
        clean_input_name = self._clean_company_name(company_name)
        
        base_ticker = None
        for key, value in self.ticker_map.items():
            if key == clean_input_name:
                base_ticker = value
                break
        
        if not base_ticker:
            logger.warning(f"Company not found in map: {company_name}")
            return None
        
        # Check if the fully qualified ticker exists in the price collection
        for suffix in ['.NS', '.BO']:
            full_ticker = f"{base_ticker}{suffix}"
            if self.price_col.find_one({'ticker': full_ticker}):
                return full_ticker
        
        logger.warning(f"No price data found for {company_name} on NSE or BSE")
        return None

    def _fetch_price_df(self, ticker: str) -> Optional[pd.DataFrame]:
        """Fetch price history from MongoDB and cache it as a DataFrame."""
        if ticker in self._price_cache:
            return self._price_cache[ticker]

        doc = self.price_col.find_one({'ticker': ticker}, {'history': 1})
        if not doc or not doc.get('history'):
            return None
        
        df = pd.DataFrame(doc['history']).set_index('date')
        df.index = pd.to_datetime(df.index)
        
        if 'close' in df.columns:
            df['close'] = pd.to_numeric(df['close'], errors='coerce')
        
        self._price_cache[ticker] = df.sort_index()
        return self._price_cache[ticker]

    # ------------------ CONSOLIDATED LLM RANKING ------------------

    def _calculate_llm_ranking_metrics(self, company_name: str, event_date_str: str, insight: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculates a single 'ranking_score' by instructing the LLM to perform a 
        comprehensive assessment of fundamentals (insight) AND price reaction (data snapshot).
        All intermediate quantitative metrics (CAR, Beta) are inferred and used internally by the LLM 
        to determine the final score, but are not returned.
        """
        ticker = self._get_ticker_by_company_name(company_name)
        
        FALLBACK_LLM_METRICS = {
            'ranking_score': 0.0,
            'esm_ticker': ticker or 'N/A'
        }
        
        if not ticker:
            logger.warning(f"LLM Ranking Fallback: No ticker found for {company_name}")
            return FALLBACK_LLM_METRICS
            
        stock_df = self._fetch_price_df(ticker)
        market_df = self._fetch_price_df(MARKET_TICKER)

        # Skip price data handling if data is missing, but still attempt ranking based on qualitative data
        has_price_data = False
        if stock_df is not None and market_df is not None and not stock_df.empty and not market_df.empty:
            
            # 1. Prepare Returns Data (Essential for price context)
            stock_df['Stock_Return'] = stock_df['close'].pct_change()
            market_df['Market_Return'] = market_df['close'].pct_change()
            
            df = pd.merge(stock_df[['Stock_Return']], market_df[['Market_Return']], left_index=True, right_index=True).dropna()
            
            if not df.empty:
                event_dt = pd.to_datetime(event_date_str)

                # 2. Extract Estimation/Event Window Data
                estimation_end_dates = df.index[df.index < event_dt]
                if not estimation_end_dates.empty:
                    estimation_end = estimation_end_dates.max()
                    estimation_df = df.loc[:estimation_end].tail(ESTIMATION_WINDOW) 
                    
                    event_start = event_dt - timedelta(days=EVENT_WINDOW_DAYS // 2)
                    event_window_df = df.loc[event_start:event_dt + timedelta(days=EVENT_WINDOW_DAYS // 2)]
                    
                    if len(estimation_df) >= 100 and not event_window_df.empty:
                         has_price_data = True

        # Package data for LLM
        prompt_data = {
            'company': company_name,
            'event_date': event_date_str,
            'fundamentals': insight.get('metrics', {}),
            'sentiment_score_finbert': insight.get('sentiment_score', 0),
            'key_highlights': insight.get('metrics', {}).get('key_highlights', []),
            'price_data_snapshot': {
                'available': has_price_data,
                'estimation_period_stats': {
                    'stock_mean_return': estimation_df['Stock_Return'].mean() if has_price_data else 'N/A',
                    'market_mean_return': estimation_df['Market_Return'].mean() if has_price_data else 'N/A',
                    'correlation': estimation_df['Stock_Return'].corr(estimation_df['Market_Return']) if has_price_data else 'N/A'
                },
                'event_period_returns_table': event_window_df.to_json(orient='table', index=True) if has_price_data else 'N/A',
                'event_days': len(event_window_df) if has_price_data else 0,
            }
        }
        
        llm_prompt = (
            "You are an expert financial analyst. Your task is to generate a single composite ranking score "
            "based on a complete assessment of the earnings event. "
            "You must integrate the company's financial fundamentals, analyst sentiment, AND the stock's price reaction "
            "during the event window (if price data is available).\n\n"
            
            "**Data Provided:**\n"
            f"{json.dumps(prompt_data, indent=2)}\n\n"
            
            "**Instructions:**\n"
            "1. **Comprehensive Assessment:** Analyze the fundamentals (revenue growth, PAT, EBITDA) and the sentiment. If 'price_data_snapshot.available' is true, analyze the price reaction. If the stock reacted positively (relative to the market) to good news, this increases the score. If price data is unavailable or unreliable, base the score *only* on fundamentals and sentiment.\n"
            "2. **ranking_score:** Generate a single 'ranking_score' between **-1.0 (Worst Event/Investment)** and **1.0 (Best Event/Investment)**. This score is a composite reflection of the event's overall investment merit.\n\n"
            
            "**Output Format (STRICTLY adhere to this JSON format only. Do not include any explanations or extra text or fields other than 'ranking_score'):**\n"
            "```json\n"
            "{\n"
            "   \"ranking_score\": <float value between -1 and 1>\n"
            "}\n"
            "```"
        )
        
        try:
            response = self.gemini_client.models.generate_content(
                model=self.GEMINI_MODEL,
                contents=llm_prompt
            )
            time.sleep(1.5) 
            
            # Extract JSON from the response text
            json_text = response.text.strip().replace('```json', '').replace('```', '').strip()
            llm_metrics = json.loads(json_text)

            # Validate and return the LLM results
            return {
                'ranking_score': float(llm_metrics.get('ranking_score', 0.0)),
                'esm_ticker': ticker
            }
        except Exception as e:
            logger.error(f"Gemini LLM Ranking failed for {company_name}: {e}. Returning fallback values.")
            return FALLBACK_LLM_METRICS


    # ------------------ LLM-Based Comparison & Ranking Rationale ------------------

    def _get_comparison_rationale(self, results: List[Dict[str, Any]]) -> str:
        """Uses LLM to provide a comparative analysis of the top-ranked events."""
        if not results:
            return "No rankable events found for comparison."

        # Pass only the critical data points for the top 10 to the LLM
        top_data = results[:10]
        
        comparison_data = [
            {
                'rank': r['rank'],
                'company': r['company'],
                'event_date': r['event_date'],
                'Ranking Score': f"{r['ranking_score']:.4f}",
                'Revenue YoY Change %': r.get('revenue_yoy_change', 'N/A'),
                'Key Highlights': r.get('key_highlights', [])[:1] # Keep only 1 highlight
            } for r in top_data
        ]
        
        prompt = (
            "Analyze the following list of top financial event results, ranked by the Composite Ranking Score. "
            "The Ranking Score integrates the company's fundamental performance, news sentiment, and the market's price reaction to the news. "
            "Provide a concise summary (under 5 sentences) of why the top-ranked company achieved the highest score. "
            "Focus your analysis on the combination of its Revenue YoY Change and Key Highlights compared to the other top-ranked companies."
            f"Data for Comparison (Top {len(comparison_data)}):\n{json.dumps(comparison_data, indent=2)}"
        )

        try:
            r = self.gemini_client.models.generate_content(
                model=self.GEMINI_MODEL,
                contents=prompt
            )
            time.sleep(1.5)
            return r.text.strip()
        except Exception as e:
            logger.error(f"Gemini comparison rationale failed: {e}")
            return "Failed to generate comparative rationale using LLM."


    def run_bulk_analysis(self, start_date_str: str, end_date_str: str) -> List[Dict[str, Any]]:
        time_cutoff = datetime.strptime(start_date_str, "%Y-%m-%d").strftime("%Y-%m-%d")
        
        insights_cursor = self.insights_col.find({
            "date": {"$gte": time_cutoff, "$lte": end_date_str}
        }).sort("date", -1)
        
        total_insights = self.insights_col.count_documents({
            "date": {"$gte": time_cutoff, "$lte": end_date_str}
        })
        logger.info(f"Found {total_insights} total insights in date range {start_date_str} to {end_date_str}")
        
        ranking_results = []
        processed_count = 0
        skipped_count = 0
        
        for insight in insights_cursor:
            processed_count += 1
            event_date = insight.get('date')
            company_name = insight.get('company')
            
            if not event_date or not company_name:
                logger.warning(f"Skipping insight {insight.get('_id')}: missing date or company")
                skipped_count += 1
                continue
                
            sentiment = insight.get('sentiment_score', 0)
            if sentiment is None:
                sentiment = 0
            
            # Single call for the composite ranking score
            llm_metrics = self._calculate_llm_ranking_metrics(company_name, event_date, insight)
            ranking_score = llm_metrics['ranking_score']
            
            # Update insight, removing old fields
            self.insights_col.update_one(
                {'_id': insight['_id']},
                {'$set': {
                    'ranking_score': ranking_score,
                    'esm_ticker': llm_metrics['esm_ticker']
                },
                '$unset': { # Explicitly remove old fields
                    'esm_car': "", 
                    'composite_score': "",
                    'esm_alpha': "",
                    'esm_beta': "",
                    'car_period_days': ""
                }}
            )
            logger.info(f"Composite Ranking calculated (LLM) for {company_name}: Ranking={ranking_score:.4f}")
            
            ranking_results.append({
                'company': company_name,
                'ticker': llm_metrics.get('esm_ticker', self._get_ticker_by_company_name(company_name)), 
                'event_date': event_date,
                'quarter': insight.get('metrics', {}).get('quarter'),
                'ranking_score': float(f"{ranking_score:.6f}"),
                'sentiment': sentiment,
                # Flatten metrics data for easy dashboard consumption
                'revenue_current': insight.get('metrics', {}).get('revenue_current_qtr'),
                'pat_current': insight.get('metrics', {}).get('pat_current_qtr'),
                'ebitda_current': insight.get('metrics', {}).get('ebitda_current_qtr'),
                'revenue_yoy_change': insight.get('metrics', {}).get('revenue_yoy_change_pct'),
                'key_highlights': insight.get('metrics', {}).get('key_highlights', [])
            })
        
        # Sort results and assign ranks
        ranking_results.sort(key=lambda x: x['ranking_score'], reverse=True)
        for i, result in enumerate(ranking_results):
            result['rank'] = i + 1
        
        # --- LLM-BASED COMPARISON ---
        comparison_rationale = self._get_comparison_rationale(ranking_results)
        logger.info(f"\n--- Comparative Ranking Rationale ---\n{comparison_rationale}\n-----------------------------------")
        
        logger.info(f"Bulk Composite Ranking analysis complete. Processed {processed_count} insights, skipped {skipped_count}, found {len(ranking_results)} rankable events.")
        
        return ranking_results


# ===================== Utility Functions =====================

def esm_already_exists_for_today():
    """Check if ranking analysis has been done for today - checks for ranking_score presence"""
    try:
        client = get_mongo_client()
        db = client[DB_NAME]
        insights_col = db[INSIGHTS_COLLECTION]
        
        today = datetime.now().strftime("%Y-%m-%d")
        existing = insights_col.find_one({
            "date": today,
            "ranking_score": {"$exists": True}
        })
        
        client.close()
        return existing is not None
    except Exception as e:
        logger.error(f"Error checking if Ranking exists for today: {e}")
        return False

def run_esm_backfill(start_date: str, end_date: str):
    price_manager = StockPriceManager()
    logger.info(f"Starting BULK BACKFILL from {start_date} to {end_date}")
    
    price_manager.backfill_history(MARKET_TICKER, start_date=start_date, end_date=end_date)
    
    for ticker in price_manager._get_all_tickers():
        price_manager.backfill_history(ticker, start_date=start_date, end_date=end_date)
        
    price_manager.client.close()
    logger.info("BULK BACKFILL complete.")

def run_esm_daily_update(daily_date):
    price_manager = StockPriceManager()
    logger.info(f"Running DAILY PRICE UPDATE for {daily_date}")
    
    if isinstance(daily_date, str):
        daily_date = datetime.strptime(daily_date, "%Y-%m-%d")

    daily_3d_ago = daily_date - timedelta(days=3)
    
    # Update market ticker first
    price_manager.backfill_history(MARKET_TICKER, start_date=daily_3d_ago, end_date=daily_date)
    
    # Update all stock tickers
    for ticker in price_manager._get_all_tickers():
        try:
            price_manager.backfill_history(ticker, start_date=daily_3d_ago, end_date=daily_date)
        except Exception as e:
            logger.error(f"Error updating {ticker}: {e}")
            continue
            
    price_manager.client.close()
    logger.info("DAILY PRICE UPDATE complete.")

def run_esm_bulk_analysis(start_date: str, end_date: str):
    esm = ESMCalculator()
    logger.info(f"Running BULK COMPOSITE RANKING ANALYSIS for insights from {start_date} to {end_date}")
    
    results = esm.run_bulk_analysis(start_date, end_date)
    
    logger.info(f"Bulk Composite Ranking analysis produced {len(results)} rankable events.")
    
    if results:
        top_results = results[:10]
        logger.info(f"Top 10 Rankings:\n{json.dumps(top_results, indent=2)}") 
        
    esm.client.close()
    return results


# ===================== Main =====================

def main():
    parser = argparse.ArgumentParser(description="BSE ESM Pipeline Coordinator")
    parser.add_argument('mode', choices=['backfill', 'daily_price_update', 'bulk_esm_analysis'], 
                         help="Mode of operation")
    parser.add_argument('--start', type=str, help="Start date for backfill/analysis (YYYY-MM-DD)")
    parser.add_argument('--end', type=str, help="End date for backfill/analysis (YYYY-MM-DD). Defaults to today")
    
    args = parser.parse_args()
    
    end_date = args.end if args.end else date_to_string(datetime.now())
    
    if args.mode in ['backfill', 'daily_price_update']:
        price_manager = StockPriceManager()
        
        if args.mode == 'backfill':
            start_date = args.start if args.start else date_to_string(datetime.now() - timedelta(days=40))
            
            logger.info(f"Running backfill from {start_date} to {end_date}")
            run_esm_backfill(start_date, end_date)
            
        elif args.mode == 'daily_price_update':
            today_str = date_to_string(datetime.now())
            
            if not esm_already_exists_for_today():
                run_esm_daily_update(today_str)
            else:
                logger.info("Ranking analysis already exists for today, skipping daily update")
        
        price_manager.client.close()
    
    elif args.mode == 'bulk_esm_analysis':
        start_date = args.start if args.start else date_to_string(datetime.now() - timedelta(days=30))
        logger.info(f"Running bulk Composite Ranking analysis for {start_date} to {end_date}")
        run_esm_bulk_analysis(start_date, end_date)

if __name__ == "__main__":
    main()