from pdf_parser import PDFProcessor

processor = PDFProcessor()

ann_id = "692abdca9b7c7a46f08fbc25"
pdf_url = "https://nsearchives.nseindia.com/corporate/SELetter_signed_11012024155319.pdf"

result = processor.process_pdf(ann_id, pdf_url)
print(result)
