import json, os, re, logging, torch, torch.nn.functional as F
from pymongo import MongoClient
from datetime import datetime
from typing import Dict, Any, Optional, List
from pydantic import BaseModel
from bson.objectid import ObjectId
from huggingface_hub import InferenceClient
from transformers import AutoTokenizer, AutoModelForSequenceClassification

os.environ["HF_TOKEN"] = "HF_TOKEN"
logger = logging.getLogger("llm_analyzer")

FINBERT_MODEL = "ProsusAI/finbert"
FINBERT_MAX_LENGTH = 512
LLM_MODEL_ID = "openai/gpt-oss-120b"

FINBERT_READY = False
try:
    FINBERT_TOKENIZER = AutoTokenizer.from_pretrained(FINBERT_MODEL)
    FINBERT_MODEL_CLASS = AutoModelForSequenceClassification.from_pretrained(FINBERT_MODEL)
    FINBERT_MODEL_CLASS = FINBERT_MODEL_CLASS.cuda() if torch.cuda.is_available() else FINBERT_MODEL_CLASS.cpu()
    FINBERT_READY = True
except:
    FINBERT_READY = False


class FinancialMetrics(BaseModel):
    revenue_current_qtr: Optional[float] = None
    revenue_ly_qtr: Optional[float] = None
    revenue_prev_qtr: Optional[float] = None
    profit_current_qtr: Optional[float] = None
    revenue_yoy_change_pct: Optional[float] = None
    revenue_qoq_change_pct: Optional[float] = None
    key_highlights: List[str] = []
    sentiment_score_finbert: float = 0.0
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
    tok = FINBERT_TOKENIZER(text, truncation=True, max_length=FINBERT_MAX_LENGTH, return_tensors="pt")
    if torch.cuda.is_available():
        tok = {k: v.cuda() for k, v in tok.items()}
    with torch.no_grad():
        out = FINBERT_MODEL_CLASS(**tok)
    p = torch.softmax(out.logits, dim=-1)
    return round(p[:, 0].item() - p[:, 1].item(), 4)


class LLMAnalyzer:
    def __init__(self, mongo_uri="mongodb://localhost:27017/", db_name="bse_data"):
        self.client = InferenceClient(base_url="https://router.huggingface.co", token=os.environ["HF_TOKEN"])
        db = MongoClient(mongo_uri)[db_name]
        self.announcements = db["announcements"]
        self.parsed_pdfs = db["parsed_pdfs"]
        self.insights = db["insights"]
        self.insights.create_index("announcement_id", unique=True)

    def get_structured_output_from_llm(self, text: str, sentiment: float, title: str) -> Optional[Dict[str, Any]]:
        prompt = f"""You are a financial data extraction expert for Indian stock market companies. You will receive unstructured text extracted from a financial-results PDF (OCR output).
The layout may be inconsistent, tables may be broken, column names may vary, and the order of information may differ.

Your task is to extract key metrics in JSON format.

[CONTENT]
Title: {title}

{text[:15000]}

[OUTPUT]
{{
  "revenue_current_qtr": "number",
  "revenue_ly_qtr": "number",
  "revenue_prev_qtr": "number",
  "profit_current_qtr": "number",
  "revenue_yoy_change_pct": "number",
  "revenue_qoq_change_pct": "number",
  "key_highlights": ["string", "string", "string"],
  "sentiment_score_finbert": "number",
  "quarter": "string"
}}

Return only valid JSON.
"""

        try:
            r = self.client.chat.completions.create(
                model=LLM_MODEL_ID,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"}
            )

            raw = r.choices[0].message["content"]

            try:
                parsed = json.loads(raw)
            except:
                parsed = json.loads(repair_json(raw))

            allowed = FinancialMetrics.model_fields.keys()
            filtered = {k: v for k, v in parsed.items() if k in allowed}
            filtered["sentiment_score_finbert"] = sentiment
            return FinancialMetrics.model_validate(filtered).model_dump()
        except:
            return None

    def analyze_document(self, parsed_pdf_id: str):
        pdf = self.parsed_pdfs.find_one({"_id": ObjectId(parsed_pdf_id)})
        if not pdf or not pdf.get("text"):
            return None
        ann_ids = pdf.get("announcement_ids", [])
        if not ann_ids:
            return None
        ann = self.announcements.find_one({"_id": ObjectId(ann_ids[0])})
        if not ann:
            return None

        metrics = self.get_structured_output_from_llm(pdf["text"], 0.0, ann.get("title", ""))
        if not metrics:
            return None

        highlights = " ".join(metrics.get("key_highlights", []))
        metrics["sentiment_score_finbert"] = get_finbert_sentiment(highlights)

        doc = {
            "announcement_id": ann_ids[0],
            "parsed_pdf_id": parsed_pdf_id,
            "company": ann.get("company"),
            "date": ann.get("date"),
            "announcement_title": ann.get("title"),
            "pdf_url": pdf.get("pdf_url"),
            "metrics": metrics,
            "analyzed_at": datetime.now()
        }

        r = self.insights.update_one({"announcement_id": ann_ids[0]}, {"$set": doc}, upsert=True)
        return r.upserted_id or r.modified_count

    def analyze_unanalyzed_pdfs(self, limit=10):
        out = []
        for pdf in self.parsed_pdfs.find({"text": {"$exists": True}}).limit(limit):
            pid = str(pdf["_id"])
            if self.insights.find_one({"parsed_pdf_id": pid}):
                continue
            res = self.analyze_document(pid)
            out.append({"parsed_pdf_id": pid, "company": pdf.get("company"), "result": "success" if res else "failed"})
        return out

    def get_insights_for_company(self, c: str):
        return list(self.insights.find({"company": c}).sort("date", -1))

    def get_recent_insights(self, limit=20):
        return list(self.insights.find().sort("analyzed_at", -1).limit(limit))
