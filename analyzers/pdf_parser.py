import os
import time
import httpx
from datetime import datetime
from pymongo import MongoClient
from bson.objectid import ObjectId
import logging
import re
import pymupdf
import pytesseract
import cv2
import numpy as np
from PIL import Image
import io
import hashlib

logger = logging.getLogger("pdf_processor")

class TextPreProcessor:
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
        self.text = re.sub(r"\n[a-zA-Z0-9]\n", "\n", self.text)
        self.text = re.sub(r"\[", "(", self.text)

    def _fix_tables(self):
        self.text = re.sub(r"\n\s*(\d{1,3}(?:,\d{3})*\.\d{2})\s*\n", r" \1 ", self.text)
        self.text = re.sub(r"\n\s*(\d{1,3}(?:,\d{3})*\.\d{2})\s*\(", r" (\1", self.text)
        self.text = re.sub(r"(\d)\n(\d)", r"\1 \2", self.text)

    def _group_paragraphs(self):
        self.text = re.sub(r"\n\s*\n\s*\n+", "||P||", self.text)
        self.text = self.text.replace("\n", " ")
        self.text = self.text.replace("||P||", "\n\n")
        self.text = re.sub(r"\s{2,}", " ", self.text).strip()

class PDFProcessor:
    def __init__(self, mongo_uri="mongodb://localhost:27017/", db_name="bse_data"):
        self.client = MongoClient(mongo_uri)
        self.db = self.client[db_name]
        self.ann = self.db["announcements"]
        self.parsed = self.db["parsed_pdfs"]
        self.parsed.create_index("pdf_hash", unique=True)
        self.parsed.create_index("announcement_id")

    def _download(self, url):
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
            "Accept": "*/*",
            "Referer": "https://www.bseindia.com/",
        }
        for attempt in range(3):
            try:
                with httpx.Client(headers=headers, timeout=60, http2=False) as client:
                    r = client.get(url)
                    r.raise_for_status()
                    return r.content
            except Exception as e:
                logger.error(f"download failed {url} attempt {attempt+1}: {e}")
                time.sleep(2)
        return None

    def _preprocess_image(self, image):
        gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
        den = cv2.fastNlMeansDenoising(gray, h=10, templateWindowSize=7, searchWindowSize=21)
        _, th = cv2.threshold(den, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return Image.fromarray(th)

    def _is_table(self, text):
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
            if not text.strip() or self._is_table(text):
                pix = page.get_pixmap(dpi=300)
                img = Image.open(io.BytesIO(pix.tobytes("png")))
                proc = self._preprocess_image(img)
                text = pytesseract.image_to_string(proc, lang="eng")
            full_text += f"\n--- Page {i+1} ---\n{text}"
        doc.close()
        return full_text

    def _consolidate(self, txt):
        return txt.strip()

    def process_pdf(self, ann_id: str, url: str):
        try:
            ann = self.ann.find_one({"_id": ObjectId(ann_id)})
            if not ann:
                return {"status": "failed", "reason": "not found"}
            pdf = self._download(url)
            if not pdf:
                return {"status": "failed", "reason": "download"}
            h = hashlib.md5(pdf).hexdigest()
            existing = self.parsed.find_one({"pdf_hash": h})
            if existing:
                self.parsed.update_one({"_id": existing["_id"]}, {"$addToSet": {"announcement_ids": ann_id}})
                return {"status": "success", "id": str(existing["_id"]), "method": "cached"}
            raw = self._extract_text_or_ocr(pdf)
            if not raw.strip():
                return {"status": "failed", "reason": "empty"}
            cleaner = TextPreProcessor(raw)
            cleaned = cleaner.run()
            final = self._consolidate(cleaned)
            try:
                d = datetime.strptime(ann.get("date", ""), "%Y-%m-%d")
            except:
                d = None
            doc = {
                "announcement_ids": [ann_id],
                "pdf_url": url,
                "pdf_hash": h,
                "text": final,
                "original_text_length": len(raw),
                "processed_at": datetime.now(),
                "text_length": len(final),
                "company": ann.get("company"),
                "announcement_date": d,
                "announcement_title": ann.get("title"),
                "category": ann.get("category"),
            }
            res = self.parsed.insert_one(doc)
            self.ann.update_one({"_id": ObjectId(ann_id)}, {"$set": {"pdf_text_id": str(res.inserted_id), "pdf_processed_status": "SUCCESS"}})
            return {"status": "success", "id": str(res.inserted_id)}
        except Exception as e:
            logger.error(f"process error {url}: {e}")
            return {"status": "failed", "reason": str(e)}

    def process_batch(self, limit=50):
        out = []
        q = self.ann.find({"pdf_url": {"$exists": True, "$ne": None, "$ne": ""}, "pdf_text_id": {"$exists": False}}).limit(limit)
        c = 0
        for ann in q:
            r = self.process_pdf(str(ann["_id"]), ann["pdf_url"])
            if r.get("status") == "success":
                c += 1
            out.append({"announcement_id": str(ann["_id"]), "company": ann.get("company"), "result": r})
            if c >= limit:
                break
        return out
