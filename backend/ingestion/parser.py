"""PDF parsing: pypdf primary, pdfplumber fallback for empty/short pages."""
from pathlib import Path

import pypdf
import structlog

log = structlog.get_logger(__name__)

# Threshold below which we consider primary extraction "failed" and try fallback.
MIN_CHARS_FOR_PRIMARY = 50


def parse_pdf(path: str | Path) -> list[tuple[int, str]]:
    """Extract page-numbered text from a PDF.

    Returns 1-indexed (page_number, text) tuples, one per page.
    Empty pages (after fallback) are returned with empty text — never crashes
    on a single bad page.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"PDF not found: {p}")

    pages: list[tuple[int, str]] = []
    reader = pypdf.PdfReader(str(p))
    for i, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception as e:
            log.warning("pypdf_extract_failed", page=i, error=str(e))
            text = ""

        pages.append((i, text))

    return _apply_pdfplumber_fallbacks_bulk(p, pages)


def _apply_pdfplumber_fallbacks_bulk(path: Path, pages: list[tuple[int, str]]) -> list[tuple[int, str]]:
    """For pages where pypdf returned little text, try pdfplumber once per file.

    Opening the PDF with pdfplumber for every weak page is catastrophically slow
    on large filings (hundreds of opens). A single open covers all weak pages.
    """
    if not any(len(t.strip()) < MIN_CHARS_FOR_PRIMARY for _, t in pages):
        return pages

    try:
        import pdfplumber
    except ModuleNotFoundError:
        return pages

    try:
        with pdfplumber.open(str(path)) as pdf:
            n = len(pdf.pages)
            out: list[tuple[int, str]] = []
            for pn, primary in pages:
                if len(primary.strip()) >= MIN_CHARS_FOR_PRIMARY or pn - 1 >= n:
                    out.append((pn, primary))
                    continue
                try:
                    fallback = pdf.pages[pn - 1].extract_text() or ""
                except Exception as e:
                    log.warning("pdfplumber_page_extract_failed", page=pn, error=str(e))
                    out.append((pn, primary))
                    continue
                if len(fallback.strip()) > len(primary.strip()):
                    log.info("pdfplumber_fallback_used", page=pn)
                    out.append((pn, fallback))
                else:
                    out.append((pn, primary))
            return out
    except Exception as e:
        log.warning("pdfplumber_bulk_fallback_failed", error=str(e))
        return pages
