"""
Diagnostic script to check MongoDB data and pipeline status
"""

from pymongo import MongoClient
from datetime import datetime
import json

def diagnose_database(mongo_uri="mongodb://localhost:27017/", db_name="bse_data"):
    """Check database collections and data structure."""
    
    client = MongoClient(mongo_uri)
    db = client[db_name]
    
    print("="*70)
    print("BSE SCRAPER DATABASE DIAGNOSTIC")
    print("="*70)
    print(f"\nDatabase: {db_name}")
    print(f"Connection: {mongo_uri}")
    print()
    
    # Check all collections
    collections = db.list_collection_names()
    print(f"Collections found: {collections}")
    print()
    
    # Check announcements collection
    print("-" * 70)
    print("ANNOUNCEMENTS COLLECTION")
    print("-" * 70)
    
    announcements = db["announcements"]
    total_announcements = announcements.count_documents({})
    print(f"Total announcements: {total_announcements}")
    
    if total_announcements > 0:
        # Check structure of first document
        sample = announcements.find_one()
        print(f"\nSample document structure:")
        print(f"  Fields: {list(sample.keys())}")
        print(f"  Company: {sample.get('company', 'N/A')}")
        print(f"  Date: {sample.get('date', 'N/A')}")
        print(f"  Title: {sample.get('title', 'N/A')[:50]}..." if sample.get('title') else "  Title: N/A")
        print(f"  PDF URL: {sample.get('pdf_url', 'N/A')}")
        
        # Check how many have PDF URLs
        with_pdf = announcements.count_documents({"pdf_url": {"$exists": True, "$ne": None, "$ne": ""}})
        print(f"\nAnnouncements with PDF URL: {with_pdf}")
        
        # Show a few examples with PDFs
        if with_pdf > 0:
            print(f"\nSample announcements with PDFs:")
            for i, doc in enumerate(announcements.find({"pdf_url": {"$exists": True, "$ne": None, "$ne": ""}}).limit(3), 1):
                print(f"\n  {i}. Company: {doc.get('company')}")
                print(f"     Date: {doc.get('date')}")
                print(f"     PDF: {doc.get('pdf_url')[:60]}...")
    else:
        print("⚠️  No announcements found!")
        print("   Make sure your BSE scraper has run and populated this collection.")
    
    # Check parsed_pdfs collection
    print("\n" + "-" * 70)
    print("PARSED_PDFS COLLECTION")
    print("-" * 70)
    
    parsed_pdfs = db["parsed_pdfs"]
    total_parsed = parsed_pdfs.count_documents({})
    print(f"Total parsed PDFs: {total_parsed}")
    
    if total_parsed > 0:
        sample = parsed_pdfs.find_one()
        print(f"\nSample parsed PDF:")
        print(f"  Company: {sample.get('company')}")
        print(f"  Text length: {sample.get('text_length')} chars")
        print(f"  Method: {sample.get('extraction_method')}")
        print(f"  Processed: {sample.get('processed_at')}")
    
    # Check insights collection
    print("\n" + "-" * 70)
    print("INSIGHTS COLLECTION")
    print("-" * 70)
    
    insights = db["insights"]
    total_insights = insights.count_documents({})
    print(f"Total insights: {total_insights}")
    
    if total_insights > 0:
        sample = insights.find_one()
        print(f"\nSample insight:")
        print(f"  Company: {sample.get('company')}")
        print(f"  Analyzed: {sample.get('analyzed_at')}")
        if sample.get('metrics'):
            print(f"  Metrics:")
            for key, value in sample['metrics'].items():
                if key != 'key_highlights':
                    print(f"    {key}: {value}")
    
    # Pipeline readiness check
    print("\n" + "="*70)
    print("PIPELINE READINESS")
    print("="*70)
    
    ready_for_pdf = announcements.count_documents({
        "pdf_url": {"$exists": True, "$ne": None, "$ne": ""}
    })
    
    unprocessed_pdfs = ready_for_pdf - parsed_pdfs.count_documents({})
    
    ready_for_analysis = parsed_pdfs.count_documents({
        "text": {"$exists": True}
    })
    
    unanalyzed = ready_for_analysis - insights.count_documents({})
    
    print(f"✓ Announcements ready for PDF processing: {ready_for_pdf}")
    print(f"✓ Unprocessed PDFs: {unprocessed_pdfs}")
    print(f"✓ PDFs ready for LLM analysis: {ready_for_analysis}")
    print(f"✓ Unanalyzed PDFs: {unanalyzed}")
    
    # Recommendations
    print("\n" + "="*70)
    print("RECOMMENDATIONS")
    print("="*70)
    
    if total_announcements == 0:
        print("❌ Run your BSE scraper first to populate announcements")
    elif ready_for_pdf == 0:
        print("❌ No announcements have PDF URLs")
        print("   Check your scraper - it should extract 'Attachment' or 'PDF' links")
    elif unprocessed_pdfs > 0:
        print(f"✅ Ready to process {unprocessed_pdfs} PDFs")
        print(f"   Run: python tasks.py --mode pdf-only --limit {min(unprocessed_pdfs, 10)}")
    elif unanalyzed > 0:
        print(f"✅ Ready to analyze {unanalyzed} PDFs")
        print(f"   Run: python tasks.py --mode analyze-only --limit {min(unanalyzed, 10)}")
    else:
        print("✅ Pipeline is up to date!")
        print("   Run ESM analysis: python esm_analyzer.py --mode batch")
    
    print("\n" + "="*70)
    
    client.close()


def show_sample_announcement_query():
    """Show how to manually query announcements."""
    
    print("\n" + "="*70)
    print("MANUAL QUERY EXAMPLES")
    print("="*70)
    
    query_examples = """
# Connect to MongoDB
from pymongo import MongoClient
client = MongoClient("mongodb://localhost:27017/")
db = client["bse_data"]

# Find announcements with PDFs
announcements_with_pdf = db.announcements.find({
    "pdf_url": {"$exists": True, "$ne": None}
}).limit(5)

for ann in announcements_with_pdf:
    print(ann['company'], ann['pdf_url'])

# Check specific company
company_docs = db.announcements.find({
    "company": {"$regex": "RELIANCE", "$options": "i"}
})

for doc in company_docs:
    print(doc)
"""
    
    print(query_examples)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Diagnose BSE pipeline database")
    parser.add_argument("--mongo-uri", default="mongodb://localhost:27017/")
    parser.add_argument("--db-name", default="bse_data")
    parser.add_argument("--show-queries", action="store_true", 
                       help="Show example MongoDB queries")
    
    args = parser.parse_args()
    
    diagnose_database(args.mongo_uri, args.db_name)
    
    if args.show_queries:
        show_sample_announcement_query()