"""MediLens AI medical RAG chatbot.

Terminal mode always uses English. Streamlit can pass "en", "hi", or "kn"
to answer_question() and receives an answer in that selected language.
"""

import json
import os
import re

import chromadb
from dotenv import load_dotenv
from google import genai
from google.genai import types


# 1. Configuration
load_dotenv()

api_key = os.getenv("GEMINI_API_KEY", "").strip().strip("'\"")
client = genai.Client(api_key=api_key) if api_key else None

# Keep the model configurable, so it can be changed without editing this file.
MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
MAX_OUTPUT_TOKENS = 240


# 2. ChromaDB
chroma_client = chromadb.PersistentClient(path="./chroma_data")
collection = chroma_client.get_or_create_collection(name="medical_knowledge")


# 3. Translated, prewritten fallback answers for the four starter topics
FALLBACK_KNOWLEDGE = {
    "doc_glucose": {
        "en": (
            "Fasting blood glucose is commonly considered normal at 70–99 mg/dL. "
            "Values of 100–125 mg/dL may indicate prediabetes. A result of "
            "126 mg/dL or higher can indicate diabetes, but diagnosis requires "
            "appropriate confirmation by a healthcare professional."
        ),
        "hi": (
            "उपवास के दौरान रक्त में ग्लूकोज़ का सामान्य स्तर आमतौर पर "
            "70–99 mg/dL होता है। 100–125 mg/dL का स्तर प्रीडायबिटीज का "
            "संकेत हो सकता है। 126 mg/dL या उससे अधिक स्तर की पुष्टि उचित "
            "जाँच और स्वास्थ्य विशेषज्ञ की सलाह से की जानी चाहिए।"
        ),
        "kn": (
            "ಉಪವಾಸದ ರಕ್ತದ ಗ್ಲೂಕೋಸ್ ಸಾಮಾನ್ಯವಾಗಿ 70–99 mg/dL ಇರುತ್ತದೆ. "
            "100–125 mg/dL ಮಟ್ಟವು ಪೂರ್ವ ಮಧುಮೇಹವನ್ನು ಸೂಚಿಸಬಹುದು. "
            "126 mg/dL ಅಥವಾ ಹೆಚ್ಚಿನ ಮಟ್ಟವನ್ನು ಸೂಕ್ತ ಪರೀಕ್ಷೆ ಮತ್ತು "
            "ಆರೋಗ್ಯ ತಜ್ಞರ ಸಲಹೆಯ ಮೂಲಕ ದೃಢಪಡಿಸಬೇಕು."
        ),
    },
    "doc_hemoglobin": {
        "en": (
            "Adult hemoglobin reference ranges vary by laboratory and individual "
            "factors. Low hemoglobin may indicate anemia and should be interpreted "
            "with other test results and clinical information."
        ),
        "hi": (
            "वयस्कों में हीमोग्लोबिन की सामान्य सीमा प्रयोगशाला और व्यक्ति के "
            "अनुसार बदल सकती है। कम हीमोग्लोबिन एनीमिया का संकेत हो सकता है। "
            "इसे अन्य जाँच परिणामों और स्वास्थ्य संबंधी जानकारी के साथ समझना चाहिए।"
        ),
        "kn": (
            "ವಯಸ್ಕರಲ್ಲಿ ಹಿಮೋಗ್ಲೋಬಿನ್‌ನ ಸಾಮಾನ್ಯ ಮಿತಿ ಪ್ರಯೋಗಾಲಯ ಮತ್ತು ವ್ಯಕ್ತಿಯ "
            "ಸ್ಥಿತಿಗೆ ಅನುಗುಣವಾಗಿ ಬದಲಾಗಬಹುದು. ಕಡಿಮೆ ಹಿಮೋಗ್ಲೋಬಿನ್ ರಕ್ತಹೀನತೆಯನ್ನು "
            "ಸೂಚಿಸಬಹುದು. ಇದನ್ನು ಇತರ ಪರೀಕ್ಷಾ ಫಲಿತಾಂಶಗಳೊಂದಿಗೆ ಅರ್ಥೈಸಬೇಕು."
        ),
    },
    "doc_creatinine": {
        "en": (
            "Creatinine is a waste product filtered by the kidneys. A high result "
            "can have several causes and should be interpreted with the laboratory "
            "range, kidney-function tests, and clinical information."
        ),
        "hi": (
            "क्रिएटिनिन एक अपशिष्ट पदार्थ है जिसे गुर्दे रक्त से छानते हैं। "
            "इसका बढ़ा हुआ स्तर कई कारणों से हो सकता है। परिणाम को प्रयोगशाला "
            "की सीमा और अन्य जाँचों के साथ समझना चाहिए।"
        ),
        "kn": (
            "ಕ್ರಿಯೇಟಿನೈನ್ ಮೂತ್ರಪಿಂಡಗಳು ರಕ್ತದಿಂದ ಶೋಧಿಸುವ ತ್ಯಾಜ್ಯ ಪದಾರ್ಥವಾಗಿದೆ. "
            "ಇದರ ಹೆಚ್ಚಿದ ಮಟ್ಟಕ್ಕೆ ಹಲವು ಕಾರಣಗಳಿರಬಹುದು. ಫಲಿತಾಂಶವನ್ನು ಪ್ರಯೋಗಾಲಯದ "
            "ಮಿತಿ ಮತ್ತು ಇತರ ಪರೀಕ್ಷೆಗಳೊಂದಿಗೆ ಅರ್ಥೈಸಬೇಕು."
        ),
    },
    "doc_cholesterol": {
        "en": (
            "Total cholesterol is one measure of blood cholesterol. Interpret it "
            "alongside the complete lipid profile and other cardiovascular risk factors."
        ),
        "hi": (
            "कुल कोलेस्ट्रॉल रक्त में कोलेस्ट्रॉल का एक माप है। इसे पूरी लिपिड "
            "प्रोफ़ाइल और हृदय संबंधी अन्य जोखिम कारकों के साथ समझना चाहिए।"
        ),
        "kn": (
            "ಒಟ್ಟು ಕೊಲೆಸ್ಟ್ರಾಲ್ ರಕ್ತದಲ್ಲಿನ ಕೊಲೆಸ್ಟ್ರಾಲ್‌ನ ಒಂದು ಮಾಪನವಾಗಿದೆ. ಇದನ್ನು "
            "ಸಂಪೂರ್ಣ ಲಿಪಿಡ್ ಪ್ರೊಫೈಲ್ ಮತ್ತು ಇತರ ಹೃದಯ ಸಂಬಂಧಿತ ಅಪಾಯದ ಅಂಶಗಳೊಂದಿಗೆ "
            "ಅರ್ಥೈಸಬೇಕು."
        ),
    },
}


# 4. Starter reference documents
MEDICAL_DOCUMENTS = [
    (
        "Fasting blood glucose is commonly considered normal at 70–99 mg/dL. "
        "Values between 100 and 125 mg/dL may indicate prediabetes. A fasting "
        "result of 126 mg/dL or higher can indicate diabetes when confirmed "
        "using appropriate testing."
    ),
    (
        "Hemoglobin is a protein in red blood cells that carries oxygen. Adult "
        "reference ranges vary by laboratory and individual factors. Low "
        "hemoglobin may indicate anemia and requires clinical interpretation."
    ),
    (
        "Creatinine is a waste product from muscle metabolism that is filtered "
        "by the kidneys. Elevated levels may indicate impaired kidney filtration "
        "but must be interpreted with other clinical findings."
    ),
    (
        "Total cholesterol measures cholesterol in the blood. Results should be "
        "interpreted with the complete lipid profile and the person's other "
        "cardiovascular risk factors."
    ),
]

# These labels are existing project metadata, not independently verified citations.
MEDICAL_METADATA = [
    {"source": "NIH MedlinePlus", "topic": "Blood Glucose", "page": 1, "doc_id": "MP-GLU-01"},
    {"source": "CDC Laboratory Reference", "topic": "Hemoglobin", "page": 3, "doc_id": "CDC-HEM-02"},
    {"source": "WHO Clinical Guidelines", "topic": "Serum Creatinine", "page": 7, "doc_id": "WHO-CRE-03"},
    {
        "source": "National Heart, Lung, and Blood Institute",
        "topic": "Cholesterol",
        "page": 2,
        "doc_id": "NHLBI-CHOL-04",
    },
]

DOCUMENT_IDS = [
    "doc_glucose",
    "doc_hemoglobin",
    "doc_creatinine",
    "doc_cholesterol",
]

_KB_INITIALIZED = False


def initialize_knowledge_base():
    """Insert or update the starter reference documents once per process."""
    global _KB_INITIALIZED
    if _KB_INITIALIZED:
        return

    collection.upsert(
        documents=MEDICAL_DOCUMENTS,
        metadatas=MEDICAL_METADATA,
        ids=DOCUMENT_IDS,
    )
    _KB_INITIALIZED = True

# 5. Language handling and short standard messages
def normalize_language(language):
    value = str(language or "en").strip().lower()
    if value in ("hi", "hindi"):
        return "hi", "Hindi"
    if value in ("kn", "kannada"):
        return "kn", "Kannada"
    return "en", "English"


MESSAGES = {
    "out_of_scope": {
        "en": (
            "I'm a medical-information chatbot, so I can help with healthcare "
            "questions only. Please ask a medical or health-related question."
        ),
        "hi": (
            "मैं चिकित्सा जानकारी देने वाला चैटबॉट हूँ और केवल स्वास्थ्य या "
            "चिकित्सा से जुड़े प्रश्नों में मदद कर सकता हूँ। कृपया ऐसा प्रश्न पूछें।"
        ),
        "kn": (
            "ನಾನು ವೈದ್ಯಕೀಯ ಮಾಹಿತಿಗಾಗಿ ಇರುವ ಚಾಟ್‌ಬಾಟ್. ಆರೋಗ್ಯ ಅಥವಾ ವೈದ್ಯಕೀಯ "
            "ಪ್ರಶ್ನೆಗಳಿಗೆ ಮಾತ್ರ ಸಹಾಯ ಮಾಡಬಹುದು. ದಯವಿಟ್ಟು ಅಂತಹ ಪ್ರಶ್ನೆಯನ್ನು ಕೇಳಿ."
        ),
    },
    "no_evidence": {
        "en": (
            "This is a medical question, but the available local references do not "
            "contain enough information for a reliable answer. Please consult a "
            "qualified healthcare professional or use a trusted medical source."
        ),
        "hi": (
            "यह चिकित्सा से संबंधित प्रश्न है, लेकिन उपलब्ध संदर्भों में "
            "विश्वसनीय उत्तर के लिए पर्याप्त जानकारी नहीं है। कृपया किसी योग्य "
            "स्वास्थ्य विशेषज्ञ या विश्वसनीय चिकित्सा स्रोत से सलाह लें।"
        ),
        "kn": (
            "ಇದು ವೈದ್ಯಕೀಯ ಪ್ರಶ್ನೆಯಾಗಿದೆ. ಆದರೆ ಲಭ್ಯವಿರುವ ಉಲ್ಲೇಖಗಳಲ್ಲಿ ವಿಶ್ವಾಸಾರ್ಹ "
            "ಉತ್ತರಕ್ಕೆ ಸಾಕಷ್ಟು ಮಾಹಿತಿ ಇಲ್ಲ. ದಯವಿಟ್ಟು ಅರ್ಹ ಆರೋಗ್ಯ ತಜ್ಞರನ್ನು ಅಥವಾ "
            "ವಿಶ್ವಾಸಾರ್ಹ ವೈದ್ಯಕೀಯ ಮೂಲವನ್ನು ಸಂಪರ್ಕಿಸಿ."
        ),
    },
    "service_unavailable": {
        "en": (
            "The answer service is temporarily unavailable, and no sufficiently "
            "relevant local reference was found. Please try again later or consult "
            "a healthcare professional."
        ),
        "hi": (
            "उत्तर सेवा अस्थायी रूप से उपलब्ध नहीं है और कोई पर्याप्त संबंधित "
            "स्थानीय संदर्भ नहीं मिला। कृपया बाद में प्रयास करें या किसी स्वास्थ्य "
            "विशेषज्ञ से सलाह लें।"
        ),
        "kn": (
            "ಉತ್ತರ ಸೇವೆ ತಾತ್ಕಾಲಿಕವಾಗಿ ಲಭ್ಯವಿಲ್ಲ ಮತ್ತು ಸಾಕಷ್ಟು ಸಂಬಂಧಿತ ಸ್ಥಳೀಯ "
            "ಉಲ್ಲೇಖ ಸಿಗಲಿಲ್ಲ. ದಯವಿಟ್ಟು ನಂತರ ಪ್ರಯತ್ನಿಸಿ ಅಥವಾ ಆರೋಗ್ಯ ತಜ್ಞರನ್ನು ಸಂಪರ್ಕಿಸಿ."
        ),
    },
}

# Shared project safety text is kept in English to preserve the existing contract.
SAFETY_MESSAGE = (
    "MediLens AI provides health information and does not replace "
    "professional medical advice."
)


# 6. Best-effort offline classification (used only if Gemini fails)
MEDICAL_TERMS = {
    "medical", "health", "symptom", "symptoms", "disease", "illness",
    "doctor", "hospital", "patient", "medicine", "medication", "treatment",
    "diagnosis", "pain", "fever", "cough", "headache", "dizzy", "dizziness",
    "tired", "fatigue", "breathing", "rash", "blood", "glucose", "sugar",
    "diabetes", "hemoglobin", "anemia", "creatinine", "kidney", "cholesterol",
    "heart", "pregnancy", "infection", "allergy", "allergic", "vomiting",
    "nausea", "stomach", "chest", "wound", "injury", "dose", "healthcare",
    "fasting", "ग्लूकोज़", "मधुमेह", "दर्द", "बुखार", "खांसी", "सिरदर्द",
    "दवा", "डॉक्टर", "स्वास्थ्य", "खून", "हीमोग्लोबिन", "किडनी",
    "ನೋವು", "ಜ್ವರ", "ಕೆಮ್ಮು", "ಔಷಧಿ", "ವೈದ್ಯರು", "ಆರೋಗ್ಯ", "ರಕ್ತ", "ಮಧುಮೇಹ",
}


def offline_medical_guess(query, distance=None):
    """Approximate medical classification if the model cannot be reached."""
    normalized = query.lower()
    if any(term in normalized for term in MEDICAL_TERMS):
        return True
    # Similarity is only a weak hint; use a strict threshold.
    return distance is not None and distance <= 0.8


def _result(status, query, answer, sources=None, insufficient=True, warnings=None):
    """Return the stable result shape expected by the Streamlit integration."""
    return {
        "status": status,
        "query": query,
        "answer": answer,
        "sources": sources or [],
        "insufficient_evidence": insufficient,
        "safety_message": SAFETY_MESSAGE,
        "warnings": warnings or [],
    }


def _source_for_streamlit(item):
    """Provide both the current source fields and the app.py-compatible title."""
    return {
        "title": item.get("source", "Unknown source"),
        "source": item.get("source", "Unknown source"),
        "source_id": item.get("id", ""),
        "topic": item.get("topic", "Unknown topic"),
        "page": item.get("page"),
        "excerpt": str(item.get("text", ""))[:160],
    }


# 7. Main RAG function
def answer_question(query: str, language: str = "en") -> dict:
    """Answer a medical question in exactly the requested language."""
    query = str(query or "").strip()
    lang_code, target_language = normalize_language(language)

    if not query:
        return _result(
            "error", query, MESSAGES["no_evidence"][lang_code],
            warnings=["Please provide a valid question."],
        )

    try:
        initialize_knowledge_base()
        count = collection.count()
        if count:
            # Three short passages are enough for the starter knowledge base and
            # reduce prompt size. Chroma's first query can still be slower while
            # its local embedding model is being initialized.
            retrieved = collection.query(
                query_texts=[query],
                n_results=min(3, count),
                include=["documents", "metadatas", "distances"],
            )
            documents = (retrieved.get("documents") or [[]])[0] or []
            metadatas = (retrieved.get("metadatas") or [[]])[0] or []
            distances = (retrieved.get("distances") or [[]])[0] or []
            ids = (retrieved.get("ids") or [[]])[0] or []
        else:
            documents, metadatas, distances, ids = [], [], [], []
    except Exception:
        return _result(
            "error", query, MESSAGES["service_unavailable"][lang_code],
            warnings=["Medical reference retrieval failed."],
        )

    context_items = []
    for index, document in enumerate(documents):
        metadata = metadatas[index] if index < len(metadatas) else {}
        context_items.append({
            "id": ids[index] if index < len(ids) else "",
            "text": document,
            "source": metadata.get("source", "Unknown source"),
            "topic": metadata.get("topic", "Unknown topic"),
            "page": metadata.get("page"),
            "distance": distances[index] if index < len(distances) else None,
        })

    # One compact Gemini request handles both scope classification and answering.
    warning = "GEMINI_API_KEY is not configured."
    if client is not None:
        compact_context = [
            {
                "text": item["text"],
                "source": item["source"],
                "topic": item["topic"],
                "page": item["page"],
                "distance": item["distance"],
            }
            for item in context_items
        ]
        prompt = f"""You are MediLens AI, a concise medical-information assistant.
Reply only in {target_language}. Do not include translations or a second language.
Classify the question by meaning:
- medical: health, healthcare, the human body, symptoms, disease, medicine, tests or prevention
- non_medical: clearly unrelated to medicine or healthcare
- uncertain: unclear intent

Retrieved local reference passages (may be irrelevant):
{json.dumps(compact_context, ensure_ascii=False)}

User question: {query}

Return only JSON with exactly these fields:
"intent": "medical", "non_medical", or "uncertain"
"context_relevant": true or false
"answer": a concise answer in {target_language}

Rules:
- If non-medical, politely say this chatbot handles medical information only.
- Answer medical questions helpfully, but do not claim unsupported facts came from the local passages.
- If local passages are insufficient, say so and give only cautious general information when appropriate.
- Do not diagnose, prescribe a medicine/dose, invent citations, or invent test results.
- Recommend urgent professional assessment for potentially emergency symptoms.
- Keep the answer short and clear. Use only {target_language}.
"""
        try:
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.1,
                    max_output_tokens=MAX_OUTPUT_TOKENS,
                    response_mime_type="application/json",
                ),
            )
            raw = (response.text or "").strip()
            model_result = json.loads(raw)
            intent = model_result.get("intent")
            answer_text = str(model_result.get("answer", "")).strip()
            context_relevant = model_result.get("context_relevant") is True

            if intent == "non_medical":
                return _result(
                    "success", query, MESSAGES["out_of_scope"][lang_code],
                    insufficient=False,
                )

            if intent == "uncertain" or intent not in ("medical", "non_medical") or not answer_text:
                return _result(
                    "partial", query, MESSAGES["no_evidence"][lang_code],
                    warnings=["The question could not be classified confidently."],
                )

            # Only show local sources when the model says the passages support the answer.
            sources = [_source_for_streamlit(item) for item in context_items] if context_relevant else []
            warnings = [] if context_relevant else [
                "The answer is general information and is not fully supported by the local knowledge base."
            ]
            return _result(
                "success" if context_relevant else "partial",
                query,
                answer_text,
                sources=sources,
                insufficient=not context_relevant,
                warnings=warnings,
            )

        except Exception as err:
            error_text = str(err).lower()
            if "429" in error_text or "quota" in error_text or "resource_exhausted" in error_text:
                warning = "Gemini quota is exhausted or rate-limited."
            else:
                warning = "Gemini is temporarily unavailable or returned an unreadable response."
    # No API key or Gemini failure: use only the small local fallback set.
    best_distance = context_items[0]["distance"] if context_items else None
    if not offline_medical_guess(query, best_distance):
        return _result(
            "partial", query, MESSAGES["out_of_scope"][lang_code],
            warnings=[warning, "Offline question classification is approximate."],
        )

    if context_items and best_distance is not None and best_distance <= 1.5:
        item = context_items[0]
        fallback_text = FALLBACK_KNOWLEDGE.get(item["id"], {}).get(lang_code)
        if fallback_text:
            prefix = {
                "en": "General information from the available local reference: ",
                "hi": "उपलब्ध स्थानीय संदर्भ से सामान्य जानकारी: ",
                "kn": "ಲಭ್ಯವಿರುವ ಸ್ಥಳೀಯ ಉಲ್ಲೇಖದಿಂದ ಸಾಮಾನ್ಯ ಮಾಹಿತಿ: ",
            }[lang_code]
            return _result(
                "partial", query, prefix + fallback_text,
                sources=[_source_for_streamlit(item)],
                insufficient=False,
                warnings=[warning, "A prewritten fallback was used because Gemini was unavailable."],
            )

    return _result(
        "partial", query, MESSAGES["no_evidence"][lang_code],
        warnings=[warning, "The local knowledge base lacks sufficient relevant evidence."],
    )


# 8. Standalone terminal test: deliberately English only.
if __name__ == "__main__":
    question = input("Ask a medical question (English): ").strip()
    if not question:
        question = "Why is my fasting blood glucose 135?"

    result = answer_question(question, language="en")
    print("\nStatus:", result["status"])
    print("Answer:", result["answer"])
    if result.get("sources"):
        print("Sources:")
        for source in result["sources"]:
            page = f", page {source['page']}" if source.get("page") is not None else ""
            print(f"- {source['title']}{page}")
    if result.get("warnings"):
        print("Warnings:", "; ".join(result["warnings"]))