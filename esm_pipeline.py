import os
import time
import httpx
import pandas as pd
import numpy as np
import statsmodels.api as sm
import yfinance as yf
from pymongo import MongoClient, UpdateOne
from pymongo.errors import BulkWriteError
from bson.objectid import ObjectId
from datetime import datetime, timedelta
import logging
import math
from typing import List, Dict, Any, Optional, Tuple
import argparse 

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
    
    def __init__(self):
        self.client = get_mongo_client()
        self.db = self.client[DB_NAME]
        self.price_col = self.db[PRICE_COLLECTION]
        self.price_col.create_index("ticker", unique=True)
        
    def _get_all_tickers(self) -> List[str]:
        if not os.path.exists(TICKER_CSV_PATH):
            logger.error(f"TICKER CSV not found at {TICKER_CSV_PATH}")
            return []

        try:
            df = pd.read_csv(TICKER_CSV_PATH)
            ticker_col = "TICKER"
            tickers = df[ticker_col].astype(str).tolist()
            return [t for t in tickers if not t.startswith('nan')]
        except Exception as e:
            logger.error(f"Error reading ticker CSV: {e}")
            return []

    def _fetch_with_exchange_fallback(self, ticker: str, start_date: str, end_date: str) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
        bse_ticker = f"{ticker}.BO"
        try:
            df = yf.download(bse_ticker, start=start_date, end=end_date, 
                            progress=False, auto_adjust=True)
            if not df.empty:
                logger.info(f"Found data for {bse_ticker} (BSE)")
                return df, bse_ticker
        except Exception as e:
            logger.debug(f"BSE fetch failed for {bse_ticker}: {e}")
        
        nse_ticker = f"{ticker}.NS"
        try:
            df = yf.download(nse_ticker, start=start_date, end=end_date, 
                            progress=False, auto_adjust=True)
            if not df.empty:
                logger.info(f"Found data for {nse_ticker} (NSE)")
                return df, nse_ticker
        except Exception as e:
            logger.debug(f"NSE fetch failed for {nse_ticker}: {e}")
        
        logger.warning(f"No data found on NSE or BSE for ticker: {ticker}")
        return None, None

    def backfill_history(self, ticker: str, start_date: str, end_date: str):
        try:
            if ticker == MARKET_TICKER:
                yahoo_ticker = MARKET_TICKER
                col = self.db.market_data 
                df = yf.download(yahoo_ticker, start=start_date, end=end_date, 
                               progress=False, auto_adjust=True)
                if df.empty:
                    logger.warning(f"No historical data for {yahoo_ticker}")
                    return
            else:
                df, yahoo_ticker = self._fetch_with_exchange_fallback(ticker, start_date, end_date)
                
                if df is None or df.empty:
                    return
                
                col = self.price_col

            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(1)
            
            price_history = []
            for idx in range(len(df)):
                try:
                    row_date = df.index[idx]
                    date_str = date_to_string(row_date) if isinstance(row_date, pd.Timestamp) else str(row_date)
                    
                    close_val = df['Close'].iloc[idx]
                    volume_val = df['Volume'].iloc[idx]
                    
                    if pd.notna(close_val):
                        price_history.append({
                            'date': date_str,
                            'close': float(close_val),
                            'volume': int(volume_val) if pd.notna(volume_val) else 0
                        })
                except Exception as e:
                    logger.warning(f"Skipping row {idx} for {yahoo_ticker}: {e}")
                    continue
            
            if not price_history:
                logger.warning(f"No valid price data extracted for {yahoo_ticker}")
                return
            
            col.update_one(
                {'ticker': yahoo_ticker},
                {'$set': {'history': price_history, 'base_ticker': ticker if ticker != MARKET_TICKER else None}},
                upsert=True
            )
            logger.info(f"Backfilled {len(price_history)} days for {yahoo_ticker}")

        except Exception as e:
            logger.error(f"Error during backfill for {ticker}: {e}")

    def run_daily_update(self, date_str: str):
        tickers = self._get_all_tickers()
        
        self.backfill_history(MARKET_TICKER, date_str, date_str) 
        
        for ticker in tickers:
            try:
                df, yahoo_ticker = self._fetch_with_exchange_fallback(ticker, date_str, date_str)
                
                if df is None or df.empty:
                    continue

                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.droplevel(1)

                close_val = df['Close'].iloc[-1]
                volume_val = df['Volume'].iloc[-1]
                row_date = df.index[-1]
                
                if pd.notna(close_val):
                    self.price_col.update_one(
                        {'ticker': yahoo_ticker},
                        {
                            '$push': {'history': {
                                'date': date_to_string(row_date) if isinstance(row_date, pd.Timestamp) else str(row_date),
                                'close': float(close_val),
                                'volume': int(volume_val) if pd.notna(volume_val) else 0
                            }}
                        },
                        upsert=True
                    )
                
            except Exception as e:
                logger.warning(f"Failed daily update for {ticker}: {e}")
        
        logger.info(f"Daily price update completed for {len(tickers)} tickers")


class ESMCalculator:

    def __init__(self):
        self.client = get_mongo_client()
        self.db = self.client[DB_NAME]
        self.price_col = self.db[PRICE_COLLECTION]
        self.insights_col = self.db[INSIGHTS_COLLECTION]
        self.market_col = self.db.market_data 
        self.ticker_map = self._load_ticker_map()

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
        
        nse_ticker = f"{base_ticker}.NS"
        if self.price_col.find_one({'ticker': nse_ticker}):
            return nse_ticker
        
        bse_ticker = f"{base_ticker}.BO"
        if self.price_col.find_one({'ticker': bse_ticker}):
            return bse_ticker
        
        logger.warning(f"No price data found for {company_name} on NSE or BSE")
        return None

    def _prepare_data_for_regression(self, ticker: str, event_date_str: str) -> Optional[pd.DataFrame]:
        stock_data = self.price_col.find_one({'ticker': ticker}, {'history': 1})
        market_data = self.market_col.find_one({'ticker': MARKET_TICKER}, {'history': 1})
        
        if not stock_data or not market_data:
            logger.warning(f"Missing history for {ticker} or {MARKET_TICKER}")
            return None
            
        stock_df = pd.DataFrame(stock_data['history']).set_index('date')
        market_df = pd.DataFrame(market_data['history']).set_index('date')
        
        stock_df.index = pd.to_datetime(stock_df.index)
        market_df.index = pd.to_datetime(market_df.index)
        
        stock_df['Stock_Return'] = stock_df['close'].pct_change()
        market_df['Market_Return'] = market_df['close'].pct_change()
        
        df = pd.merge(stock_df[['Stock_Return']], market_df[['Market_Return']], left_index=True, right_index=True).dropna()
        return df

    def get_car_for_event(self, company_name: str, event_date_str: str) -> Optional[Dict[str, float]]:
        ticker = self._get_ticker_by_company_name(company_name)
        if not ticker:
            return None
            
        df = self._prepare_data_for_regression(ticker, event_date_str)
        if df is None or df.empty:
            return None

        event_dt = pd.to_datetime(event_date_str)

        estimation_end = df.index[df.index < event_dt].max()
        estimation_df = df.loc[:estimation_end].tail(ESTIMATION_WINDOW).iloc[:-1] 
        
        if len(estimation_df) < 100:
             logger.warning(f"ESM skipped: Insufficient estimation data ({len(estimation_df)} days)")
             return None

        X = sm.add_constant(estimation_df['Market_Return'])
        Y = estimation_df['Stock_Return']
        model = sm.OLS(Y, X).fit()
        alpha = model.params['const']
        beta = model.params['Market_Return']

        event_start = event_dt - timedelta(days=EVENT_WINDOW_DAYS // 2)
        event_window_df = df.loc[event_start:event_dt + timedelta(days=EVENT_WINDOW_DAYS // 2)]
        
        if event_window_df.empty:
            return None

        event_window_df['Expected_Return'] = alpha + beta * event_window_df['Market_Return']
        event_window_df['Abnormal_Return'] = event_window_df['Stock_Return'] - event_window_df['Expected_Return']
        car = event_window_df['Abnormal_Return'].sum()
        
        return {
            'ticker': ticker,
            'car': float(f"{car:.6f}"),
            'alpha': float(f"{alpha:.6f}"),
            'beta': float(f"{beta:.6f}"),
            'car_period_days': len(event_window_df)
        }

    def _calculate_composite_score(self, metrics: Dict[str, Any]) -> float:
        sentiment = metrics.get('sentiment_score_finbert', 0)
        
        revenue_current = metrics.get('revenue_current_qtr')
        revenue_previous = metrics.get('revenue_previous_qtr')
        profit_current = metrics.get('profit_current_qtr')
        profit_previous = metrics.get('profit_previous_qtr')
        
        revenue_growth = 0
        if revenue_current and revenue_previous and revenue_previous != 0:
            revenue_growth = (revenue_current - revenue_previous) / abs(revenue_previous)
        
        profit_growth = 0
        if profit_current and profit_previous and profit_previous != 0:
            profit_growth = (profit_current - profit_previous) / abs(profit_previous)
        
        profit_margin = 0
        if profit_current and revenue_current and revenue_current != 0:
            profit_margin = profit_current / revenue_current
        
        composite = (
            sentiment * 0.3 +
            min(max(revenue_growth, -1), 1) * 0.25 +
            min(max(profit_growth, -1), 1) * 0.25 +
            min(max(profit_margin, -1), 1) * 0.2
        )
        
        return composite

    def run_bulk_analysis(self, start_date_str: str, end_date_str: str) -> List[Dict[str, Any]]:
        time_cutoff = datetime.strptime(start_date_str, "%Y-%m-%d").strftime("%Y-%m-%d")
        
        insights_cursor = self.insights_col.find({
            "date": {"$gte": time_cutoff, "$lte": end_date_str},
            "metrics.revenue_current_qtr": {"$ne": None} 
        }).sort("date", -1)
        
        ranking_results = []
        
        for insight in insights_cursor:
            event_date = insight['date']
            company_name = insight['company']
            metrics = insight.get('metrics', {})
            
            if insight.get('esm_car') is not None and insight.get('ranking_score') is not None:
                ranking_score = insight.get('ranking_score', 0)
                car_score = insight.get('esm_car', 0)
                composite_score = insight.get('composite_score', 0)
            else:
                esm_data = self.get_car_for_event(company_name, event_date)
                
                if not esm_data:
                    continue
                    
                car_score = esm_data['car']
                composite_score = self._calculate_composite_score(metrics)
                ranking_score = car_score * (1 + composite_score)
                
                self.insights_col.update_one(
                    {'_id': insight['_id']},
                    {'$set': {
                        'esm_car': car_score, 
                        'composite_score': composite_score,
                        'ranking_score': ranking_score
                    }}
                )
            
            ranking_results.append({
                'company': company_name,
                'ticker': self._get_ticker_by_company_name(company_name),
                'event_date': event_date,
                'car_score': car_score,
                'composite_score': composite_score,
                'ranking_score': float(f"{ranking_score:.6f}"),
                'sentiment': metrics.get('sentiment_score_finbert', 0),
                'revenue_current': metrics.get('revenue_current_qtr'),
                'profit_current': metrics.get('profit_current_qtr')
            })
        
        ranking_results.sort(key=lambda x: x['ranking_score'], reverse=True)
        logger.info(f"Bulk ESM analysis complete. Found {len(ranking_results)} rankable events")
        
        return ranking_results


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
            start_date = args.start if args.start else date_to_string(datetime.now() - timedelta(days=365))
            
            logger.info(f"Starting BULK BACKFILL from {start_date} to {end_date}")
            
            price_manager.backfill_history(MARKET_TICKER, start_date, end_date)
            
            for ticker in price_manager._get_all_tickers():
                 price_manager.backfill_history(ticker, start_date, end_date)
            
        elif args.mode == 'daily_price_update':
            daily_date = args.start if args.start else date_to_string(datetime.now() - timedelta(days=1))
            logger.info(f"Running DAILY PRICE UPDATE for {daily_date}")
            price_manager.run_daily_update(daily_date)
            
        price_manager.client.close()

    elif args.mode == 'bulk_esm_analysis':
        start_date = args.start if args.start else date_to_string(datetime.now() - timedelta(days=7)) 
        
        esm = ESMCalculator()
        logger.info(f"Running BULK ESM ANALYSIS for insights from {start_date} to {end_date}")
        
        results = esm.run_bulk_analysis(start_date, end_date)
        
        logger.info("\n=== TOP RANKED ANNOUNCEMENT EVENTS ===")
        results.sort(key=lambda x: x['ranking_score'], reverse=True)
        
        for i, res in enumerate(results[:10]):
            logger.info(f"{i+1}. {res['company']} ({res['ticker']}) | "
                       f"CAR: {res['car_score']:.2%} | "
                       f"Composite: {res['composite_score']:.3f} | "
                       f"Ranking: {res['ranking_score']:.3f}")
            
        esm.client.close()


if __name__ == "__main__":
    main()