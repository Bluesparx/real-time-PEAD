import logging
from datetime import datetime, timedelta
import asyncio

from analyzers.llm_analyzer import LLMAnalyzer
from analyzers.esm import run_esm_bulk_analysis, run_esm_daily_update, esm_already_exists_for_today
from scraper_cron import scrape_all

logger = logging.getLogger("pipeline_orchestrator")

class PipelineOrchestrator:
    def __init__(self, mongo_uri="mongodb://localhost:27017/", db_name="bse_data"):
        self.llm_analyzer = LLMAnalyzer(mongo_uri=mongo_uri, db_name=db_name)
        logger.info("Pipeline Orchestrator initialized.")

    # --- Scraper ---
    async def run_scraper(self, days_ago: int):
        logger.info(f"Starting Scraper for the last {days_ago} days.")
        try:
            end_date = datetime.now().strftime("%Y-%m-%d")
            start_date = (datetime.now() - timedelta(days=days_ago)).strftime("%Y-%m-%d")
            await scrape_all(start=start_date, end=end_date)
            logger.info("Scraper complete.")
            return {"start_date": start_date, "end_date": end_date, "status": "success"}
        except Exception as e:
            logger.error(f"Scraper failed: {e}", exc_info=True)
            return {"status": "failed", "error": str(e)}

    # --- LLM Analysis (Standalone) ---
    async def run_llm_pipeline(self, limit: int = 50):
        """
        Finds unanalyzed PDFs and runs LLM analysis on them.
        Fully independent of PDFProcessor.
        """
        logger.info(f"Running LLM pipeline on up to {limit} un-analyzed PDFs...")
        try:
            results = await asyncio.to_thread(self.llm_analyzer.analyze_unanalyzed_pdfs, limit)
            for r in results:
                logger.info(f"LLM analysis for PDF {r['parsed_pdf_id']} ({r['company']}): {r.get('result', 'failed')}")
            logger.info("LLM pipeline complete")
            return results
        except Exception as e:
            logger.error(f"Error running LLM pipeline: {e}", exc_info=True)
            return []

    # --- ESM Analysis ---
    def run_esm_analysis(self, days_ago: int):
        if esm_already_exists_for_today():
            logger.info("ESM for today already exists. Skipping.")
            return []
        logger.info("Running ESM analysis...")
        try:
            end_date = datetime.now().strftime("%Y-%m-%d")
            start_date = (datetime.now() - timedelta(days=days_ago)).strftime("%Y-%m-%d")
            results = run_esm_bulk_analysis(start_date, end_date)
            return results if results else []
        except Exception as e:
            logger.error(f"Error in ESM analysis: {e}", exc_info=True)
            return []

    # --- Full Pipeline ---
    async def run_full_pipeline(self, days_ago: int = 7, limit: int = 50, skip_scraper=False):
        logger.info("=" * 80)
        logger.info("STARTING FULL PIPELINE")
        logger.info("=" * 80)

        scraper_stats = {"status": "skipped"}

        # Step 1: Scraper
        if not skip_scraper:
            scraper_stats = await self.run_scraper(days_ago)
            logger.info(f"Scraper completed: {scraper_stats}")

        # Step 2: LLM Pipeline
        llm_results = await self.run_llm_pipeline(limit=limit)
        logger.info(f"LLM analysis complete: {len(llm_results)} PDFs processed")

        # # Step 3: Daily stock update
        # try:
        #     run_esm_daily_update(datetime.now().strftime("%Y-%m-%d"))
        #     logger.info("Stock prices updated successfully")
        # except Exception as e:
        #     logger.error(f"Error updating stock prices: {e}", exc_info=True)

        # Step 4: ESM analysis
        esm_results = self.run_esm_analysis(days_ago)
        logger.info(f"ESM analysis completed: {len(esm_results)} results")

        logger.info("=" * 80)
        logger.info("PIPELINE COMPLETE")
        logger.info("=" * 80)

        return {
            "scraper": scraper_stats,
            "llm_analysis_count": len(llm_results),
            "esm_analysis_count": len(esm_results),
            "timestamp": datetime.now()
        }