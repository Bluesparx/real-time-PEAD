import json, os, re, logging, torch, torch.nn.functional as F
from pymongo import MongoClient
from datetime import datetime
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, ValidationError 
from bson.objectid import ObjectId
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from google import genai
import time
import hashlib
import httpx
import pymupdf
import pytesseract
import cv2
import numpy as np
from PIL import Image
import io

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


# --- INTEGRATED PDF PARSING LOGIC ---

class TextPreProcessor:
    """Cleans and structures raw OCR/PDF text for optimal LLM consumption."""
    def __init__(self, raw_text: str):
        self.raw = raw_text
        self.text = self.raw

    def run(self) -> str:
        self._remove_noise()
        self._clean_lines()
        self._fix_tables()
        self._group_paragraphs()
        return self.text

    def _remove_noise(self):
        self.text = self.text.replace("--- End of Page ---", "\n*** NEW PAGE ***\n")

    def _clean_lines(self):
        # Remove standalone single character lines (often noise)
        self.text = re.sub(r"\n[a-zA-Z0-9]\n", "\n", self.text)
        self.text = re.sub(r"\[", "(", self.text)

    def _fix_tables(self):
        # Consolidate split numbers/decimals within tables
        self.text = re.sub(r"\n\s*(\d{1,3}(?:,\d{3})*\.\d{2})\s*\n", r" \1 ", self.text)
        self.text = re.sub(r"\n\s*(\d{1,3}(?:,\d{3})*\.\d{2})\s*\(", r" (\1", self.text)
        # Fix numbers split across lines (e.g., 1\n234)
        self.text = re.sub(r"(\d)\n(\d)", r"\1 \2", self.text)

    def _group_paragraphs(self):
        # Standardize triple newlines to mark paragraph breaks
        self.text = re.sub(r"\n\s*\n\s*\n+", "||P||", self.text)
        # Collapse remaining newlines into spaces
        self.text = self.text.replace("\n", " ")
        # Restore paragraph breaks
        self.text = self.text.replace("||P||", "\n\n")
        self.text = re.sub(r"\s{2,}", " ", self.text).strip()


class PDFTextProcessor:
    """Handles PDF downloading, text/OCR extraction, and storage of parsed text."""
    def __init__(self, db: MongoClient):
        self.ann = db["announcements"]
        self.parsed = db["parsed_pdfs"]
        self.parsed.create_index("pdf_hash", unique=True)
        self.parsed.create_index("announcement_id")
        self.client = httpx.Client(timeout=60, http2=False)

    def _download(self, url):
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
            "Accept": "*/*",
            "Referer": "https://www.bseindia.com/",
        }
        for attempt in range(3):
            try:
                r = self.client.get(url, headers=headers)
                r.raise_for_status()
                return r.content
            except Exception as e:
                logger.error(f"Download failed for {url} attempt {attempt+1}: {e}")
                time.sleep(2)
        return None

    def _preprocess_image(self, image):
        # Convert to numpy array and grayscale
        gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
        # Denoising
        den = cv2.fastNlMeansDenoising(gray, h=10, templateWindowSize=7, searchWindowSize=21)
        # Thresholding (Otsu's method)
        _, th = cv2.threshold(den, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return Image.fromarray(th)

    def _is_table(self, text):
        # Simple heuristic to detect table-like text (many numbers per line)
        lines = text.split("\n")
        for line in lines:
            numbers = sum(1 for w in line.split() if any(c.isdigit() for c in w))
            if numbers >= 3:
                return True
        return False

    def _extract_text_or_ocr(self, pdf_bytes):
        doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
        full_text = ""
        for i, page in enumerate(doc):
            text = page.get_text()
            # If text is empty or suspected to be a table (poor extraction), run OCR
            if not text.strip() or self._is_table(text):
                pix = page.get_pixmap(dpi=300)
                img = Image.open(io.BytesIO(pix.tobytes("png")))
                proc = self._preprocess_image(img)
                text = pytesseract.image_to_string(proc, lang="eng")
            full_text += f"\n--- Page {i+1} ---\n{text}"
        doc.close()
        return full_text

    def process_pdf(self, ann_id: str, url: str):
        """Downloads, parses (text/OCR), cleans, and stores the PDF text."""
        try:
            ann = self.ann.find_one({"_id": ObjectId(ann_id)})
            if not ann:
                return {"status": "failed", "reason": "announcement not found"}
            
            pdf = self._download(url)
            if pdf is None:
                return {"status": "failed", "reason": "download failed"}
            
            h = hashlib.md5(pdf).hexdigest()
            existing = self.parsed.find_one({"pdf_hash": h})
            
            if existing:
                # Deduplication handled: link announcement to existing parsed PDF
                self.parsed.update_one({"_id": existing["_id"]}, {"$addToSet": {"announcement_ids": ann_id}})
                return {"status": "success", "id": str(existing["_id"]), "method": "cached"}
            
            raw = self._extract_text_or_ocr(pdf)
            if not raw.strip():
                return {"status": "failed", "reason": "empty text"}
            
            # Cleaning and consolidation
            cleaner = TextPreProcessor(raw)
            cleaned = cleaner.run()
            final = cleaned.strip()

            announcement_date = datetime.strptime(ann.get("date", ""), "%Y-%m-%d") if ann.get("date") else None
            
            doc = {
                "announcement_ids": [ann_id],
                "pdf_url": url,
                "pdf_hash": h,
                "text": final,
                "original_text_length": len(raw),
                "processed_at": datetime.now(),
                "text_length": len(final),
                "company": ann.get("company"),
                "announcement_date": announcement_date,
                "announcement_title": ann.get("title"),
                "category": ann.get("category"),
            }
            res = self.parsed.insert_one(doc)
            
            # Update original announcement record
            self.ann.update_one({"_id": ObjectId(ann_id)}, {"$set": {"pdf_text_id": str(res.inserted_id), "pdf_processed_status": "SUCCESS"}})
            
            return {"status": "success", "id": str(res.inserted_id)}
            
        except Exception as e:
            logger.error(f"PDF processing error for {url}: {e}", exc_info=True)
            return {"status": "failed", "reason": str(e)}

    def process_batch(self, limit=50):
        """Finds announcements without parsed text and processes their PDFs."""
        
        # Find announcements that have a PDF URL but no associated parsed text ID
        q = self.ann.find(
            {"pdf_url": {"$exists": True, "$ne": None, "$ne": ""}, 
             "pdf_text_id": {"$exists": False}}
        ).limit(limit)
        
        out = []
        for ann in q:
            r = self.process_pdf(str(ann["_id"]), ann["pdf_url"])
            out.append({"announcement_id": str(ann["_id"]), "company": ann.get("company"), "result": r})
            
        return out

# --- END INTEGRATED PDF PARSING LOGIC ---


class LLMAnalyzer:
    def __init__(self, mongo_uri="mongodb://localhost:27017/", db_name="bse_data"):
        db = MongoClient(mongo_uri)[db_name]
        self.announcements = db["announcements"]
        self.parsed_pdfs = db["parsed_pdfs"]
        self.insights = db["insights"]
        self.insights.create_index("announcement_id", unique=True)
        
        # Initialize integrated PDF processor
        self.pdf_processor = PDFTextProcessor(db) 
        
        self.GEMINI_MODEL = "gemini-2.5-flash" 
        self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY", "AIzaSyA5dlO-h-uv5XBEuRpzBXj1l8qOzp9oyow"))

    def get_structured_output_from_llm(self, text: str, title: str) -> Optional[Dict[str, Any]]:
        prompt = f"""
You are a financial-data extraction model specialized in messy OCR text from Indian quarterly financial results.
Your primary task is to extract key financial figures based on the provided document text.

[RULES]
- Always use Rs. Lakhs (Rs. Lacs) for currency.
- Match fields by meaning (e.g., income=revenue, eps=earning per share). If no exact match is found then use the most semantically similar available value (or calculate it based on other values and financial knowledge)
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
        highlights_text = " ".join(key_highlights) if key_highlights else ""
        sentiment_score_finbert = get_finbert_sentiment(highlights_text)

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
        """
        Processes PDFs for announcements lacking parsed text, then analyzes the resulting text.
        Returns analysis results for new insights generated.
        """
        pdf_process_results = self.pdf_processor.process_batch(limit=limit)
        logger.info(f"PDF Processing Batch complete. {len([r for r in pdf_process_results if r['result']['status'] == 'success'])} PDFs parsed/cached.")
        
        analyzed_ann_ids = {doc["announcement_id"] for doc in self.insights.find({}, {"announcement_id": 1})}
        
        out = []
        
        # Use the results from the batch process to focus on recently updated announcements
        # We need to find the corresponding PDF text ID (pdf_text_id) for these announcements
        
        # Fallback query to find parsed PDFs that still need analysis
        # Find PDF documents where 'text' exists, but whose announcement_id isn't in 'analyzed_ann_ids'
        q = self.parsed_pdfs.find(
             {"text": {"$exists": True, "$ne": None}}
        ).limit(limit)
        
        for pdf in q:
            pid = str(pdf["_id"])
            
            # Check if any linked announcement ID for this PDF already has an insight
            if any(ann_id in analyzed_ann_ids for ann_id in pdf.get("announcement_ids", [])):
                continue

            res = self.analyze_document(pid)
            
            company_name = pdf.get("company", "Unknown")

            out.append({"parsed_pdf_id": pid, "company": company_name, "result": "success" if res else "failed"})
            
        return out

    def get_insights_for_company(self, c: str):
        return list(self.insights.find({"company": c}).sort("date", -1))

    def get_recent_insights(self, limit=20):
        return list(self.insights.find().sort("analyzed_at", -1).limit(limit))