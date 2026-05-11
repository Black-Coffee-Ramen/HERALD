import io
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib import colors

def generate_pdf_report(scan_record):
    """
    Generates a PDF executive summary for a domain scan record.
    Returns a BytesIO buffer.
    """
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    
    # Title
    c.setFont("Helvetica-Bold", 20)
    c.drawString(50, height - 50, "HERALD Detection Report")
    
    # Header Line
    c.setStrokeColor(colors.black)
    c.line(50, height - 60, width - 50, height - 60)
    
    # Content
    y_position = height - 100
    
    fields = [
        ("Domain:", scan_record.domain),
        ("Target Brand:", scan_record.target_cse),
        ("Detection Label:", scan_record.label),
        ("ML Confidence:", f"{scan_record.confidence:.4f}" if scan_record.confidence is not None else "N/A"),
        ("Source:", scan_record.source),
        ("Scan Date:", str(scan_record.scan_date)),
        ("Is Live:", str(scan_record.is_live)),
        ("Analyst Verdict:", str(scan_record.analyst_verdict) if scan_record.analyst_verdict else "None"),
    ]
    
    for label, value in fields:
        c.setFont("Helvetica-Bold", 12)
        c.drawString(50, y_position, label)
        c.setFont("Helvetica", 12)
        c.drawString(180, y_position, str(value))
        y_position -= 30
        
    # Warning for Phishing
    if scan_record.label == "Phishing":
        c.setFillColor(colors.red)
        c.setFont("Helvetica-Bold", 14)
        c.drawString(50, y_position - 20, "CRITICAL: This domain exhibits high-confidence phishing indicators.")
    elif scan_record.label == "Suspected":
        c.setFillColor(colors.orange)
        c.setFont("Helvetica-Bold", 14)
        c.drawString(50, y_position - 20, "WARNING: This domain requires analyst review.")
        
    c.showPage()
    c.save()
    
    buffer.seek(0)
    return buffer
