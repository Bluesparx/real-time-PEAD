import json, os, re, logging, torch, torch.nn.functional as F
from pymongo import MongoClient
from datetime import datetime
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, ValidationError # Import ValidationError for targeted catching
from bson.objectid import ObjectId
from huggingface_hub import InferenceClient
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from google import genai
import time

# Use os.getenv for token, falling back to 'hf' placeholder if not set
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN", "hf")
logger = logging.getLogger("llm_analyzer")

FINBERT_MODEL = "ProsusAI/finbert"
FINBERT_MAX_LENGTH = 512
LLM_MODEL_ID = "openai/gpt-oss-120b"

FINBERT_READY = False
try:
    FINBERT_TOKENIZER = AutoTokenizer.from_pretrained(FINBERT_MODEL)
    # Ensure model is moved to GPU/CPU only once
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    FINBERT_MODEL_CLASS = AutoModelForSequenceClassification.from_pretrained(FINBERT_MODEL).to(device)
    FINBERT_READY = True
    logger.info(f"FinBERT loaded successfully on {device}")
except Exception as e:
    FINBERT_READY = False
    logger.warning(f"FinBERT initialization failed: {e}")


class FinancialMetrics(BaseModel):
    # ... (Model Definition remains the same)
    revenue_current_qtr: Optional[float] = None
    pat_current_qtr: Optional[float] = None
    ebitda_current_qtr: Optional[float] = None
    revenue_yoy_change_pct: Optional[float] = None
    revenue_qoq_change_pct: Optional[float] = None
    eps_current_qtr: Optional[float] = None
    key_highlights: List[str] = []
    quarter: str = "Unknown"


def repair_json(s: str) -> str:
    s = re.sub(r',\s*}', '}', s)
    s = re.sub(r'"[A-Za-z0-9_]+\s*"$', '', s)
    ob, cb = s.count('{'), s.count('}')
    while cb < ob:
        s += '}'
        cb += 1
    return s


def get_finbert_sentiment(text: str) -> float:
    if not FINBERT_READY or not text.strip():
        return 0.0
    tok = FINBERT_TOKENIZER(text, truncation=True, max_length=FINBERT_MAX_LENGTH, return_tensors="pt").to(FINBERT_MODEL_CLASS.device)
    with torch.no_grad():
        out = FINBERT_MODEL_CLASS(**tok)
    p = torch.softmax(out.logits, dim=-1)
    return round(p[:, 0].item() - p[:, 1].item(), 4)


class LLMAnalyzer:
    def __init__(self, mongo_uri="mongodb://localhost:27017/", db_name="bse_data"):
        db = MongoClient(mongo_uri)[db_name]
        self.announcements = db["announcements"]
        self.parsed_pdfs = db["parsed_pdfs"]
        self.insights = db["insights"]
        self.insights.create_index("announcement_id", unique=True)
        self.GEMINI_MODEL = "gemini-2.5-flash" 
        # FIX: Use os.getenv for API key
        self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY", "gapi"))

    def get_structured_output_from_llm(self, text: str, title: str) -> Optional[Dict[str, Any]]:
        prompt = f"""
You are a financial-data extraction model specialized in messy OCR text from Indian quarterly financial results.
Your primary task is to extract key financial figures based on the provided document text.

[RULES]
- Always use Rs. Lakhs (Rs. Lacs) for currency.
- Match fields by meaning (e.g., income=revenue, eps=earning per share).  If no exact match is found then use the most semantically similar available value (or calculate it based on other values and financial knowledge)
- PAT = any number described as PAT, Net Profit, Profit After Tax, Profit for the Period or Quarter, Net Income, or Total Profit: treat all these as pat_current_qtr.
- EBITDA = any number described as EBITDA, EBITA, EBIDTA,Profit Before Tax (PBT), Operating Profit, or Operating Earnings: treat all these as ebitda_current_qtr.
- Normalize numbers: convert "1,23,4O0" to "123400"; use null if ambiguous.
- Prefer audited or clearly labeled quarterly values.
- Calculate YoY: (current_quarter - same_quarter_last_year) / same_quarter_last_year * 100.
- Calculate QoQ: (current_quarter - previous_quarter) / previous_quarter * 100.
- Extract 2-4 key business highlights (e.g., increase/decrease, new product, resignation).
- Never output negative results; percentage change can be negative.
- Output only valid JSON.
- Format quarter as Q#FY## (e.g., Q2FY24).

Extract ONLY the following fields and return VALID JSON:

{{
"revenue_current_qtr": number or null,
"pat_current_qtr": number or null,
"ebitda_current_qtr": number or null,
"revenue_yoy_change_pct": number or null,
"revenue_qoq_change_pct": number or null,
"eps_current_qtr": number or null,
"key_highlights": [string],
"quarter": string or null
}}

Analyze the following text:
[CONTENT START]
{text[:15000]}
[CONTENT END]
"""

        # FIX: Outer try block is now focused on parsing and validation, with targeted error handling
        try:
            r = None
            try:
                r = self.client.models.generate_content(
                    model=self.GEMINI_MODEL,
                    contents=prompt
                )
                time.sleep(0.46)
            except Exception as api_e:
                logger.error(f"Gemini API call failed for document '{title}': {api_e}")
                return None
            
            if r is None: return None 

            raw = r.text.strip()
            raw = raw.replace('```json', '').replace('```', '').strip()
            
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError:
                parsed = json.loads(repair_json(raw))
                logger.warning(f"JSON required repair for document '{title}'. Raw: {raw[:50]}...")
            
            allowed = FinancialMetrics.model_fields.keys()
            filtered = {k: v for k, v in parsed.items() if k in allowed}
            
            return FinancialMetrics.model_validate(filtered).model_dump()
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse repaired JSON for '{title}'. Error: {e}. Raw Output: {raw[:200]}...")
            return None
        except ValidationError as e:
            logger.error(f"Pydantic validation failed for '{title}'. Error: {e.errors()[:3]}. Parsed Data: {filtered}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error during structured output processing for '{title}': {e}")
            return None
            
            
    def analyze_document(self, parsed_pdf_id: str):
        pdf = self.parsed_pdfs.find_one({"_id": ObjectId(parsed_pdf_id)})
        if not pdf:
            logger.error(f"PDF document not found for ID: {parsed_pdf_id}")
            return None
        if not pdf.get("text"):
            logger.warning(f"PDF document ID {parsed_pdf_id} found, but 'text' field is empty.")
            return None

        ann_ids = pdf.get("announcement_ids", [])
        if not ann_ids:
            logger.warning(f"PDF ID {parsed_pdf_id} has no linked announcement IDs.")
            return None
            
        ann = self.announcements.find_one({"_id": ObjectId(ann_ids[0])})
        if not ann:
            logger.warning(f"Announcement ID {ann_ids[0]} not found for PDF ID {parsed_pdf_id}.")
            return None

        title = ann.get("title", f"Document ID {parsed_pdf_id}")
        logger.info(f"Starting analysis for company: {ann.get('company')} - {title[:50]}...")

        metrics = self.get_structured_output_from_llm(pdf["text"], title)
        if not metrics:
            logger.error(f"Failed to extract metrics for: {title}. Skipping insight creation.")
            return None
            
        key_highlights = metrics.get("key_highlights", [])
        if key_highlights:
            highlights_text = " ".join(key_highlights)
            sentiment_score_finbert = get_finbert_sentiment(highlights_text)
        else:
            sentiment_score_finbert = 0.0 

        doc = {
            "announcement_id": ann_ids[0],
            "parsed_pdf_id": parsed_pdf_id,
            "company": ann.get("company"),
            "date": ann.get("date"),
            "announcement_title": ann.get("title"),
            "pdf_url": pdf.get("pdf_url"),
            "metrics": metrics,
            "sentiment_score": sentiment_score_finbert, 
            "analyzed_at": datetime.now()
        }
        try:
            r = self.insights.update_one({"announcement_id": ann_ids[0]}, {"$set": doc}, upsert=True)
            logger.info(f"Successfully created/updated insight for {ann.get('company')} (ID: {ann_ids[0]})")
            return r.upserted_id or r.modified_count
        except Exception as db_e:
            logger.critical(f"FATAL DB WRITE ERROR for {ann.get('company')}: {db_e}")
            return None


    def analyze_unanalyzed_pdfs(self, limit=10):
        analyzed_ids = {doc["parsed_pdf_id"] for doc in self.insights.find({}, {"parsed_pdf_id": 1})}
        
        out = []
        logger.info(f"Checking for unanalyzed PDFs (Found {len(analyzed_ids)} existing insights).")
        
        for pdf in self.parsed_pdfs.find({"text": {"$exists": True}}).limit(limit):
            pid = str(pdf["_id"])
            if pid in analyzed_ids:
                continue

            res = self.analyze_document(pid)
            
            company_name = pdf.get("company", "Unknown")
            if company_name == "Unknown" and pdf.get("announcement_ids"):
                ann = self.announcements.find_one({"_id": pdf["announcement_ids"][0]})
                company_name = ann.get("company") if ann else "Unknown"

            out.append({"parsed_pdf_id": pid, "company": company_name, "result": "success" if res else "failed"})
            
        return out

    def get_insights_for_company(self, c: str):
        return list(self.insights.find({"company": c}).sort("date", -1))

    def get_recent_insights(self, limit=20):
        return list(self.insights.find().sort("analyzed_at", -1).limit(limit))