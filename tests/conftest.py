"""Shared test fixtures."""
from pathlib import Path
import pytest
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter


@pytest.fixture
def synthetic_pdf(tmp_path) -> Path:
    """3-page PDF with known content per page. Used by parser/chunker tests."""
    path = tmp_path / "synthetic.pdf"
    c = canvas.Canvas(str(path), pagesize=letter)
    pages = [
        "Page one talks about revenue: INR 12,114 crore in FY24.",
        "Page two talks about Blinkit's contribution growing 50 percent year-on-year.",
        "Page three discusses governance, risk, and compliance frameworks.",
    ]
    for text in pages:
        c.drawString(72, 720, text)
        c.showPage()
    c.save()
    return path
