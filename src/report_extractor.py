
import io
import pymupdf
import pytesseract
from PIL import Image

import os

# Configure Tesseract on Windows only when it exists.
if os.name == "nt":
    tesseract_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

    if os.path.isfile(tesseract_path):
        pytesseract.pytesseract.tesseract_cmd = tesseract_path

def extract_report(file_bytes, filename):
    source_type = (
        filename.rsplit(".", 1)[-1].lower()
        if isinstance(filename, str) and "." in filename
        else "unknown"
    )

    if source_type != "pdf":
        return {
            "status": "error",
            "text": "",
            "source_type": source_type,
            "pages_processed": 0,
            "warnings": ["Only PDF files are supported."]
        }

    if not isinstance(file_bytes, bytes) or not file_bytes:
        return {
            "status": "error",
            "text": "",
            "source_type": "pdf",
            "pages_processed": 0,
            "warnings": ["A non-empty PDF byte sequence is required."]
        }

    doc = None

    try:
        doc = pymupdf.open(stream=file_bytes, filetype="pdf")
        pages_processed = len(doc)
        text_parts = []
        warnings = []

        for page_number, page in enumerate(doc, start=1):
            page_text = page.get_text().strip()

            if page_text:
                text_parts.append(page_text)
                continue

            # No selectable text: render the page and try OCR.
            try:
                pix = page.get_pixmap(
                    matrix=pymupdf.Matrix(2, 2),
                    alpha=False
                )
                image_bytes = pix.tobytes("png")

                with Image.open(io.BytesIO(image_bytes)) as image:
                    ocr_text = pytesseract.image_to_string(
                        image,
                        lang="eng"
                    ).strip()

                if ocr_text:
                    text_parts.append(ocr_text)
                    warnings.append(
                        f"OCR was used on page {page_number}; "
                        "verify the extracted text."
                    )
                else:
                    text_parts.append("")
                    warnings.append(
                        f"No selectable text or OCR text found "
                        f"on page {page_number}."
                    )

            except Exception as exc:
                text_parts.append("")
                warnings.append(
                    f"OCR failed on page {page_number}: "
                    f"{type(exc).__name__}: {exc}"
                )

        text = "\n\n".join(text_parts).strip()

        if not text:
            return {
                "status": "partial",
                "text": "",
                "source_type": "pdf",
                "pages_processed": pages_processed,
                "warnings": warnings or [
                    "No text could be extracted from the PDF."
                ]
            }

        status = "partial" if warnings else "success"

        return {
            "status": status,
            "text": text,
            "source_type": "pdf",
            "pages_processed": pages_processed,
            "warnings": warnings
        }

    except Exception as exc:
        return {
            "status": "error",
            "text": "",
            "source_type": "pdf",
            "pages_processed": 0,
            "warnings": [
                f"PDF extraction failed: {type(exc).__name__}: {exc}"
            ]
        }

    finally:
        if doc is not None:
            doc.close()
