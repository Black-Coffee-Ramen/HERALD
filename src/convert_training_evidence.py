# convert_training_evidence.py
import os
import fitz  # PyMuPDF

EVIDENCE_DIR = "data/raw/PS02_Training_set/Evidences"
PNG_DIR = "reference/phishing_evidence"
os.makedirs(PNG_DIR, exist_ok=True)

for pdf_file in os.listdir(EVIDENCE_DIR):
    if pdf_file.endswith(".pdf"):
        pdf_path = os.path.join(EVIDENCE_DIR, pdf_file)
        png_path = os.path.join(PNG_DIR, pdf_file.replace(".pdf", ".png"))
        try:
            # Open the PDF
            pdf_document = fitz.open(pdf_path)
            
            # Get the first page
            first_page = pdf_document[0]
            
            # Convert to image
            pix = first_page.get_pixmap()
            pix.save(png_path)
            
            pdf_document.close()
            print(f"✅ Converted {pdf_file}")
        except Exception as e:
            print(f"❌ Failed {pdf_file}: {e}")