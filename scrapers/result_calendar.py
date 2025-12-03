import asyncio
import os
import time
import httpx
from datetime import datetime, timedelta
from playwright.async_api import async_playwright
from pymongo import MongoClient, UpdateOne
from pymongo.errors import BulkWriteError
import logging
import sys
from typing import List, Dict, Any

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('./logs/bse_forthcoming.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

FORTHCOMING_API_URL = "https://api.bseindia.com/BseIndiaAPI/api/Corpforthresults/w"
MAX_RETRIES = 5
TIMEOUT_SECONDS = 60
BATCH_SIZE = 100

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
DB_NAME = os.getenv("DB_NAME", "bse_data")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "result_calendar")

def get_mongo_client():
    try:
        client = MongoClient(MONGO_URI)
        client.admin.command('ping')
        logger.info("Successfully connected to MongoDB")
        return client
    except Exception as e:
        logger.error(f"Failed to connect to MongoDB: {e}")
        raise

def save_to_mongodb(mongo_client, records: List[Dict[str, Any]]):
    if not records:
        return 0
    try:
        db = mongo_client[DB_NAME]
        collection = db[COLLECTION_NAME]
        
        # Ensure unique index exists for robust deduplication
        collection.create_index([("scrip_code", 1), ("meeting_date_raw", 1)], unique=True, background=True)
        
        operations = []
        for record in records:
            # Use a composite key for upsert operation
            scrip_code = record.get("scrip_code", "")
            meeting_date_raw = record.get("meeting_date_raw", "")

            operations.append(UpdateOne(
                {"scrip_code": scrip_code, "meeting_date_raw": meeting_date_raw},
                {"$set": record},
                upsert=True
            ))
            
        result = collection.bulk_write(operations, ordered=False)
        inserted = result.upserted_count
        modified = result.modified_count
        matched = result.matched_count # Records that already existed
        total_processed = len(records)
        
        logger.info(f"MongoDB Bulk Write: Processed {total_processed} records. Inserted {inserted}, Modified {modified}, Matched {matched} existing.")
        
        return inserted + modified
    except BulkWriteError as e:
        logger.warning(f"Bulk write partially failed: {e.details}")
        # Note: If a record fails validation/index, it won't be counted in upserted/modified counts.
        return e.details.get('nUpserted', 0) + e.details.get('nModified', 0)
    except Exception as e:
        logger.error(f"Error saving to MongoDB: {e}")
        return 0

async def get_session_cookies():
    async with async_playwright() as pw:
        user_agent_override = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        browser = await pw.chromium.launch(headless=True)
        context = await browser.new_context(user_agent=user_agent_override)
        page = await context.new_page()
        await page.goto("https://www.bseindia.com/corporates/Forth_Results.html")
        await page.wait_for_load_state("networkidle")
        await asyncio.sleep(2)
        cookies = await context.cookies()
        await browser.close()
        cookie_header = "; ".join(f"{c['name']}={c['value']}" for c in cookies)
        return cookie_header, user_agent_override

def fetch_data_for_range(client, start_date_ymd, end_date_ymd):
    params = {"fromdate": start_date_ymd, "scripcode": "", "todate": end_date_ymd}
    for attempt in range(MAX_RETRIES):
        try:
            r = client.get(FORTHCOMING_API_URL, params=params)
            r.raise_for_status()
            data = r.json()
            if isinstance(data, list):
                return data
            time.sleep(2 * (attempt + 1))
        except Exception:
            time.sleep(2 * (attempt + 1))
    return []

async def scrape_forthcoming_results():
    start_date = datetime.now()
    end_date = start_date + timedelta(days=30)
    start_date_ymd = start_date.strftime("%Y%m%d")
    end_date_ymd = end_date.strftime("%Y%m%d")
    mongo_client = get_mongo_client()
    try:
        cookie_header, user_agent = await get_session_cookies()
        headers = {
            "User-Agent": user_agent,
            "Cookie": cookie_header,
            "Accept": "application/json, text/plain, */*",
            "Content-Type": "application/json",
            "Origin": "https://www.bseindia.com",
            "Referer": "https://www.bseindia.com/corporates/Forth_Results.html",
        }
        with httpx.Client(headers=headers, timeout=TIMEOUT_SECONDS) as client:
            logger.info(f"Fetching forthcoming results from {start_date_ymd} to {end_date_ymd}")
            data = fetch_data_for_range(client, start_date_ymd, end_date_ymd)
            if data:
                records = []
                for item in data:
                    meeting_date_raw = item.get("meeting_date", "")
                    try:
                        meeting_date_standard = datetime.strptime(meeting_date_raw, "%d %b %Y").strftime("%Y-%m-%d")
                    except Exception:
                        meeting_date_standard = None
                    row = {
                        "scrip_code": item.get("scrip_Code", "").strip(),
                        "company_name": item.get("Long_Name", "").strip(),
                        "TICKER": item.get("short_name", "").strip(),
                        "meeting_date_raw": meeting_date_raw,
                        "meeting_date_standard": meeting_date_standard,
                        "bse_url": item.get("URL", "").strip(),
                        "scraped_at": datetime.now()
                    }
                    records.append(row)
                save_to_mongodb(mongo_client, records)
                logger.info(f"Successfully scraped and processed {len(records)} potential result calendar entries.")
            else:
                 logger.warning("No forthcoming results data returned from API.")
    finally:
        mongo_client.close()

if __name__ == "__main__":
    asyncio.run(scrape_forthcoming_results())