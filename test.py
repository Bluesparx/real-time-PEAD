import os
from datetime import datetime
from pymongo import MongoClient
from bson.objectid import ObjectId

# Import the main analysis class
from analyzers.llm_analyzer import LLMAnalyzer 

# --- CONFIGURATION ---
MONGO_URI = "mongodb://localhost:27017/"
DB_NAME = "bse_data"

# The specific PDF URL and a placeholder Announcement ID (must be a 24-char hex string)
PDF_URL = "https://assets.airtel.in/static-assets/cms/investor/docs/quarterly_results/2025-26/Q2/Published-Results.pdf"
ANNOUNCEMENT_ID = "caaa6aa692560322B9987420" 

# --- EXECUTION ---

# 1. Initialize the analyzer and MongoDB client
analyzer = LLMAnalyzer(mongo_uri=MONGO_URI, db_name=DB_NAME)
client = MongoClient(MONGO_URI)
db = client[DB_NAME]
announcements_collection = db["announcements"]


print(f"--- Starting Analysis for Announcement ID: {ANNOUNCEMENT_ID} ---")

# --- STEP 0: SETUP - ENSURE ANNOUNCEMENT RECORD EXISTS ---
print("Step 0: Creating/Updating prerequisite announcement record...")

announcement_doc = {
    "_id": ObjectId(ANNOUNCEMENT_ID.zfill(24)), # Ensure it's a valid 24-char ObjectId
    "company": "Bharti Airtel Ltd",
    "date": "2025-11-01", # Placeholder date
    "title": "Unaudited Financial Results for Q2FY26",
    "pdf_url": PDF_URL,
    "scraped_at": datetime.now()
}

# Insert or replace the prerequisite announcement document
announcements_collection.replace_one(
    {"_id": ObjectId(ANNOUNCEMENT_ID.zfill(24))},
    announcement_doc,
    upsert=True
)

print(f"Prerequisite announcement '{ANNOUNCEMENT_ID}' ensured in 'announcements' collection.")
# --- END STEP 0 ---


# 2. Force PDF Processing (Download, OCR, Text Save) for the specific document.
# We call the internal PDFTextProcessor.process_pdf directly via the analyzer instance.
try:
    print(f"\nStep 1: Processing PDF from URL: {PDF_URL}")
    
    # We simulate the PDF processing step that usually happens in the batch loop.
    pdf_processing_result = analyzer.pdf_processor.process_pdf(ANNOUNCEMENT_ID, PDF_URL)
    
    if pdf_processing_result['status'] == 'success':
        parsed_pdf_id = pdf_processing_result['id']
        print(f"PDF Processing SUCCESS. Parsed PDF ID: {parsed_pdf_id}. Method: {pdf_processing_result.get('method', 'New')}")

        # 3. Analyze the document (LLM Extraction and Insight Creation)
        print("\nStep 2: Running LLM Analysis and Insight Creation...")
        
        # We call the core analysis method directly on the parsed PDF ID
        insight_id = analyzer.analyze_document(parsed_pdf_id)
        
        if insight_id:
            print(f"\n--- SUCCESS ---")
            print(f"Insight created/updated successfully in 'insights' collection.")
            
            # Retrieve the newly created insight to show the structured data
            new_insight = analyzer.insights.find_one({"announcement_id": ANNOUNCEMENT_ID})
            
            if new_insight:
                print(f"Company: {new_insight.get('company')}")
                print(f"Sentiment Score: {new_insight.get('sentiment_score', 'N/A')}")
                print(f"Metrics (Partial): {new_insight.get('metrics', {}).get('revenue_current_qtr', 'N/A')}")
                print(f"Check MongoDB 'insights' collection for document ID: {new_insight['_id']}")
            else:
                print("Could not retrieve new insight document.")
        else:
            print("--- FAILURE: LLM Analysis Failed or returned None. ---")

    else:
        print(f"--- FAILURE: PDF Processing Failed. Reason: {pdf_processing_result['reason']} ---")

except Exception as e:
    print(f"\nCRITICAL ERROR during script execution: {e}")
finally:
    analyzer.client.close()
    client.close()
    print("\nMongoDB connections closed.")