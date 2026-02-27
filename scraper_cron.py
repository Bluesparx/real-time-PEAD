import asyncio
import os
import time
import httpx
import math
from datetime import datetime, timedelta
from playwright.async_api import async_playwright
from pymongo import MongoClient, UpdateOne
from pymongo.errors import BulkWriteError
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('bse_scraper.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

# --- Config ---
API_URL = "https://api.bseindia.com/BseIndiaAPI/api/AnnSubCategoryGetData/w"
ANN_PER_PAGE = 50
MAX_RETRIES = 5
TIMEOUT_SECONDS = 60
BATCH_SIZE = 100 


MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
DB_NAME = os.getenv("DB_NAME", "bse_data")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "announcements")


## DB helpers
# ---------------------

def get_mongo_client():
    """Create MongoDB client connection."""
    try:
        client = MongoClient(MONGO_URI)
        client.admin.command('ping')
        logger.info("Successfully connected to MongoDB")
        return client
    except Exception as e:
        logger.error(f"Failed to connect to MongoDB: {e}")
        raise

def save_to_mongodb(mongo_client, records):
    """Save records to MongoDB using upsert to avoid duplicates."""
    if not records:
        return 0
    
    try:
        db = mongo_client[DB_NAME]
        collection = db[COLLECTION_NAME]
        
        collection.create_index(
            [("date", 1), ("company", 1), ("title", 1)],
            unique=True,
            background=True
        )
        
        operations = []
        for record in records:
            operations.append(
                UpdateOne(
                    {
                        "date": record["date"],
                        "company": record["company"],
                        "title": record["title"]
                    },
                    {"$set": record},
                    upsert=True
                )
            )
        
        result = collection.bulk_write(operations, ordered=False)
        
        inserted = result.upserted_count
        modified = result.modified_count
        
        logger.info(f"MongoDB: Inserted {inserted} new, Modified {modified} existing records in batch.")
        return inserted + modified
        
    except BulkWriteError as e:
        logger.warning(f"Bulk write partially failed: {e.details}")
        return e.details.get('nUpserted', 0) + e.details.get('nModified', 0)
    except Exception as e:
        logger.error(f"Error saving to MongoDB: {e}")
        return 0


def generate_monthly_ranges(start_date_str, end_date_str):
    """Generate monthly or partial-month date ranges in YYYYMMDD format."""
    start = datetime.strptime(start_date_str, "%Y-%m-%d")
    end = datetime.strptime(end_date_str, "%Y-%m-%d")

    ranges = []
    curr = start

    while curr <= end:
        next_month = curr.replace(day=28) + timedelta(days=4)
        month_end = next_month - timedelta(days=next_month.day)
        end_date = min(month_end, end)

        ranges.append((curr.strftime("%Y%m%d"), end_date.strftime("%Y%m%d")))
        curr = next_month.replace(day=1)

    return ranges


def calculate_start_date(months_ago=3):
    """Calculates the date 'months_ago' and returns as YYYY-MM-DD string."""
    today = datetime.now()
    start_date = today - timedelta(days=30 * months_ago)
    return start_date.strftime("%Y-%m-%d")


async def get_session_cookies():
    """Open BSE announcements page once to gather cookies and user agent."""
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)
        context = await browser.new_context()
        page = await context.new_page()

        await page.goto("https://www.bseindia.com/corporates/ann.html")
        await page.wait_for_load_state("networkidle")
        await asyncio.sleep(1.5) 

        cookies = await context.cookies()
        user_agent = await page.evaluate("() => navigator.userAgent")

        await browser.close()

        cookie_header = "; ".join(f"{c['name']}={c['value']}" for c in cookies)
        return cookie_header, user_agent


def fetch_all_announcements_for_range(client, category, start_date, end_date):
    """
    Fetches all announcements by calculating total pages (ROWCNT/50) 
    then iterating through all pages with retries.
    """
    all_data = []
    total_pages = 1

    params = {
        "pageno": "1",
        "strCat": category,
        "strPrevDate": start_date,
        "strToDate": end_date,
        "strScrip": "",
        "strSearch": "P",
        "strType": "C",
        "subcategory": "-1",
    }

    logger.info(f"Finding total count for category '{category}'.")

    for attempt in range(MAX_RETRIES):
        try:
            r = client.get(API_URL, params=params)
            r.raise_for_status()
            data = r.json()

            total_count = 0
            if isinstance(data, dict) and data.get("Table1"):
                total_count = data["Table1"][0].get("ROWCNT", 0)

            if total_count > 0:
                total_pages = math.ceil(total_count / ANN_PER_PAGE)
                logger.info(f"Total {total_count} records across {total_pages} pages.")
                break
            else:
                logger.info("No announcements found.")
                return []

        except Exception as e:
            logger.warning(f"Error retrieving total count (attempt {attempt + 1}): {e}")
            time.sleep(2 * (attempt + 1))
    else:
        logger.error("Failed to retrieve total count after maximum retries.")
        return []

    #  fetch all pages
    for page_num in range(1, total_pages + 1):
        params["pageno"] = str(page_num)
        logger.info(f"Fetching page {page_num} of {total_pages}.")

        for attempt in range(MAX_RETRIES):
            try:
                r = client.get(API_URL, params=params)
                r.raise_for_status()
                data = r.json()

                announcements = data.get("Table", [])
                if announcements:
                    all_data.extend(announcements)
                    time.sleep(1.0) 
                    break
            except Exception as e:
                logger.warning(f"Error on page {page_num}, attempt {attempt + 1}: {e}")
                time.sleep(2 * (attempt + 1))
        else:
            logger.error(f"Failed to fetch page {page_num} after retries. Continuing to next page.")

    logger.info(f"Completed pagination for '{category}' ({start_date} to {end_date}).")
    return all_data


async def scrape_all(start=None, end=None):
    """
    Main function to orchestrate the scraping and MongoDB saving.
    """
    if start is None or end is None:
        today = datetime.now().strftime("%Y-%m-%d")
        start = end = today
    
    logger.info(f"Starting scrape from {start} to {end}")
    
    mongo_client = get_mongo_client()
    
    try:
        cookie_header, user_agent = await get_session_cookies()

        headers = {
            "User-Agent": user_agent,
            "Cookie": cookie_header,
            "Accept": "application/json, text/plain, */*",
            "Content-Type": "application/json",
            "Origin": "https://www.bseindia.com",
            "Referer": "https://www.bseindia.com/corporates/ann.html",
        }

        # our model focuses on results + pdf parsing of results
        categories = [
            "Result"
        ]


        ranges = generate_monthly_ranges(start, end)

        with httpx.Client(headers=headers, timeout=TIMEOUT_SECONDS) as client:
            buffer_rows = []
            total_records = 0

            for start_d, end_d in ranges:
                logger.info(f"\nDate Range: {start_d} to {end_d}")

                for category in categories:
                    logger.info(f"Processing category: {category}")

                    data = fetch_all_announcements_for_range(client, category, start_d, end_d)
                    if not data:
                        logger.info(f"No data for category '{category}'.")
                        continue

                    for item in data:
                        title = item.get("MORE", "").strip() or item.get("HEADLINE", "").strip()
                        title = title.replace("\n", " ").replace("\r", " ")

                        attachment = item.get("ATTACHMENTNAME", "").strip()
                        
                        pdf_link = ""
                        if attachment:
                            pdf_link = f"https://www.bseindia.com/xml-data/corpfiling/AttachHis/{attachment}"

                        row = {
                            "date": item.get("DissemDT", "").split("T")[0],
                            "company": item.get("SLONGNAME", "").strip(),
                            "category": item.get("CATEGORYNAME", "").strip(),
                            "title": title,
                            "attachment": attachment, 
                            "pdf_url": pdf_link,     
                            "scraped_at": datetime.now()
                        }
                        
                        buffer_rows.append(row)
                        
                        if len(buffer_rows) >= BATCH_SIZE:
                            saved = save_to_mongodb(mongo_client, buffer_rows)
                            total_records += saved
                            buffer_rows = []

                    logger.info(f"Completed category '{category}' with {len(data)} records fetched.")

            # Save any remaining rows
            if buffer_rows:
                saved = save_to_mongodb(mongo_client, buffer_rows)
                total_records += saved

            logger.info(f"\nTotal {total_records} announcements processed and saved to MongoDB")
            
    except Exception as e:
        logger.error(f"Error during scraping: {e}")
        raise
    finally:
        mongo_client.close()
        logger.info("MongoDB connection closed")


async def scrape_today():
    """Scrape only today's announcements - for daily cron job."""
    today = datetime.now().strftime("%Y-%m-%d")
    await scrape_all(start=today, end=today)


if __name__ == "__main__":
    import sys
    
    # Example 1: python scraper_cron.py today
    # Example 2: python scraper_cron.py 2025-11-01 2025-11-30
    
    if len(sys.argv) > 1 and sys.argv[1] == "today":
        asyncio.run(scrape_today())
    elif len(sys.argv) == 3:
        asyncio.run(scrape_all(sys.argv[1], sys.argv[2]))
    else:
        logger.info("No date arguments provided. Defaulting to today's date.")
        asyncio.run(scrape_today())