""" Pipeline Orchestrator for BSE Financial Data Processing This script coordinates the entire pipeline: 
1. Fetch announcements from BSE scraper collection 
2. Process PDFs and extract text 
3. Analyze extracted text with LLM 
4. Store insights 

Usage: 
   python tasks.py --mode full --limit 10 
   python tasks.py --mode pdf-only --limit 5 
   python tasks.py --mode analyze-only --limit 5 
"""

import argparse
import logging
from datetime import datetime
from pdf_parser import PDFProcessor
from llm_analyzer import LLMAnalyzer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(f'pipeline_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("pipeline_orchestrator")


class PipelineOrchestrator:
    def __init__(self, mongo_uri="mongodb://localhost:27017/", db_name="bse_data"):
        self.pdf_processor = PDFProcessor(mongo_uri=mongo_uri, db_name=db_name)
        self.llm_analyzer = LLMAnalyzer(mongo_uri=mongo_uri, db_name=db_name)
        logger.info("Pipeline Orchestrator initialized.")

    def run_pdf_processing(self, limit=10):
        logger.info(f"Starting PDF processing (limit: {limit})")
        results = self.pdf_processor.process_batch(limit=limit)
        success = sum(1 for r in results if r["result"]["status"] == "success")
        logger.info(f"PDF processing complete: {success}/{len(results)} successful")
        return results

    def run_llm_analysis(self, limit=10):
        logger.info(f"Starting LLM analysis (limit: {limit})")
        results = self.llm_analyzer.analyze_unanalyzed_pdfs(limit=limit)
        success = sum(1 for r in results if r["result"] == "success")
        logger.info(f"LLM analysis complete: {success}/{len(results)} successful")
        return results

    def run_full_pipeline(self, limit=10):
        logger.info("STARTING FULL PIPELINE")
        pdf_results = self.run_pdf_processing(limit=limit)
        llm_results = self.run_llm_analysis(limit=limit)
        logger.info("PIPELINE COMPLETE")
        return {
            "pdf_processing": pdf_results,
            "llm_analysis": llm_results,
            "timestamp": datetime.now()
        }

    def get_pipeline_statistics(self):
        stats = {
            "announcements_total": self.pdf_processor.ann.count_documents({}),
            "announcements_with_pdf": self.pdf_processor.ann.count_documents(
                {"pdf_url": {"$exists": True, "$ne": None}}
            ),
            "parsed_pdfs_total": self.pdf_processor.parsed.count_documents({}),
            "insights_total": self.llm_analyzer.insights.count_documents({}),
            "timestamp": datetime.now()
        }
        for k, v in stats.items():
            if k != "timestamp":
                logger.info(f"{k}: {v}")
        return stats


def main():
    parser = argparse.ArgumentParser(description="BSE Financial Data Processing Pipeline")
    parser.add_argument("--mode", choices=["full", "pdf-only", "analyze-only", "stats"], default="full")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--mongo-uri", default="mongodb://localhost:27017/")
    parser.add_argument("--db-name", default="bse_data")

    args = parser.parse_args()

    orch = PipelineOrchestrator(mongo_uri=args.mongo_uri, db_name=args.db_name)

    if args.mode == "full":
        orch.run_full_pipeline(limit=args.limit)
    elif args.mode == "pdf-only":
        orch.run_pdf_processing(limit=args.limit)
    elif args.mode == "analyze-only":
        orch.run_llm_analysis(limit=args.limit)
    elif args.mode == "stats":
        orch.get_pipeline_statistics()

    logger.info("Execution complete.")


if __name__ == "__main__":
    main()
