
# MediLens AI — Medical Report Processing Module

## Overview

This module extracts text from medical PDF reports, parses
laboratory test results, and uses the Gemini API to generate
patient-friendly explanations.

## Features

- PDF text extraction using PyMuPDF
- Parsing of test names, values, units, and reference ranges
- Gemini API integration for plain-language explanations
- Basic parser unit tests
- Error handling for PDF and API failures

## Project Structure

```text
medilens-ai/
├── app.py
├── requirements.txt
├── .env
├── .gitignore
└── src/
    ├── report_extractor.py
    ├── report_parser.py
    ├── llm_service.py
    ├── test_parser.py
    ├── test_pipeline.py
    └── sample_report.pdf
```

## Setup

1. Create and activate a Python virtual environment.
2. Install dependencies:

   ```bash
   python -m pip install -r requirements.txt
   ```

3. Create a `.env` file in the project root:

   ```text
   GEMINI_API_KEY=your_api_key_here
   ```

4. Ensure `.env` is included in `.gitignore`.

## Run Parser Tests

From the `src` directory:

```bash
python -m unittest test_parser.py
```

## Run the Complete Pipeline

Place `sample_report.pdf` in the `src` directory and run:

```bash
python test_pipeline.py
```

The pipeline extracts PDF text, parses test results, and requests
a patient-friendly explanation from Gemini.

## Limitations

- The parser supports specific line-oriented report formats and
  may not handle every laboratory layout correctly.
- Scanned PDFs require OCR support, which is not yet included.
- Gemini availability depends on the selected model and API service.
- Generated explanations are informational and are not a diagnosis.
- Clinical interpretation should be performed by a qualified
  healthcare professional.
