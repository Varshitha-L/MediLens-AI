
import pymupdf


def extract_text(pdf_path):
    doc = None

    try:
        doc = pymupdf.open(pdf_path)
        pages_processed = len(doc)
        text_parts = []
        warnings = []

        for page_number, page in enumerate(doc, start=1):
            page_text = page.get_text()
            text_parts.append(page_text)

            if not page_text.strip():
                warnings.append(
                    f"No selectable text found on page {page_number}."
                )

        text = "\n".join(text_parts)

        if not text.strip():
            return {
                "status": "partial",
                "text": "",
                "source_type": "pdf",
                "pages_processed": pages_processed,
                "warnings": warnings or [
                    "No selectable text found in the PDF."
                ]
            }

        return {
            "status": "success",
            "text": text,
            "source_type": "pdf",
            "pages_processed": pages_processed,
            "warnings": warnings
        }

    except Exception as e:
        return {
            "status": "error",
            "text": "",
            "source_type": "pdf",
            "pages_processed": 0,
            "warnings": [str(e)]
        }

    finally:
        if doc is not None:
            doc.close()
