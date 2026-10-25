# MediLens AI — Member 1 Starter

A Streamlit front end and integration shell for the MediLens AI student prototype.

## Run locally (Windows)

1. Install Python 3.11 or 3.12 and VS Code.
2. Open this folder in VS Code.
3. In VS Code, choose **Terminal → New Terminal** and run:

   ```powershell
   py -m venv .venv
   .\.venv\Scripts\Activate.ps1
   python -m pip install -r requirements.txt
   python -m streamlit run app.py
   ```

4. Streamlit will show a local URL, usually `http://localhost:8501`.

If PowerShell blocks virtual-environment activation, use Command Prompt and run `.venv\Scripts\activate.bat`, or follow Python's official virtual-environment instructions.

## Module contracts for teammates

- `src/report_extractor.py`: `extract_report(file_bytes, filename)`
- `src/report_parser.py`: `parse_report(extracted_text)`
- `src/abnormality_checker.py`: `check_abnormalities(parsed_report)`
- `src/llm_service.py`: `explain_report(parsed_report, flags, language)`
- `src/medical_rag.py`: `answer_question(question, language)`
- `src/translation.py`: `translate_text(text, language_code)`
- `src/safety.py`: `validate_report(parsed_report)`

Each module must return a dictionary that includes `status` (`success`, `partial`, or `error`) and a `warnings` list. Keep the exact function names and argument order unless the team agrees and documents a change. The UI already handles modules that have not been added yet.

## GitHub and deployment

Commit the project to a GitHub repository. Once the app and dependencies work locally, deploy it at Streamlit Community Cloud by selecting the repository, branch, and `app.py` entrypoint. Add any real API key using the deployment platform's secrets manager; never put secrets in code or GitHub.

## Safety and privacy

Use synthetic or authorized, de-identified reports during development and demos. Do not upload identifiable patient reports to a public demo or commit medical reports, `.env`, or secrets to GitHub. The prototype is for health awareness, not diagnosis or treatment.
