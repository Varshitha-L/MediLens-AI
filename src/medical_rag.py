"""Evidence-grounded medical Q&A using ChromaDB retrieval and Gemini generation.

The curated source collection is intentionally transparent: each document has a
source title and a trusted public URL. Gemini must answer only from retrieved
documents. If generation is unavailable, the app displays clearly labelled
source material instead of pretending a stored passage is AI-generated.
"""
from __future__ import annotations

import os
from typing import Any

import chromadb
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

PRIMARY_MODEL = "gemini-3.8-flash"
FALLBACK_MODEL = "gemini-3.5-flash-lite"
MAX_RETRIEVED_SOURCES = 3
MAX_DISTANCE = 1.5
REQUEST_TIMEOUT_MS = 45000

chroma_client = chromadb.PersistentClient(path="./chroma_data")
collection = chroma_client.get_or_create_collection(name="medical_knowledge")
_KNOWLEDGE_BASE_READY = False

# Curated learning material from public health agencies. These are short
# educational summaries, not personal medical advice or diagnostic rules.
MEDICAL_DOCUMENTS: list[dict[str, str]] = [
    {
        "id": "doc_cbc",
        "title": "Complete Blood Count (CBC)",
        "topic": "Blood tests",
        "source": "MedlinePlus",
        "url": "https://medlineplus.gov/lab-tests/complete-blood-count-cbc/",
        "text": (
            "A complete blood count (CBC) measures several parts of the blood. "
            "Red blood cells carry oxygen; white blood cells help fight infections; "
            "platelets help blood clot; hemoglobin is the oxygen-carrying protein in "
            "red blood cells; hematocrit is the proportion of blood made up of red "
            "blood cells; and mean corpuscular volume (MCV) is the average size of "
            "red blood cells. A CBC is one tool used with symptoms, medical history "
            "and other information. An out-of-range result does not by itself prove "
            "a disease or mean treatment is needed."
        ),
    },
    {
        "id": "doc_hemoglobin",
        "title": "Hemoglobin Test",
        "topic": "Red blood cells and oxygen transport",
        "source": "MedlinePlus",
        "url": "https://medlineplus.gov/lab-tests/hemoglobin-test/",
        "text": (
            "Hemoglobin is an iron-rich protein inside red blood cells. It carries "
            "oxygen from the lungs to the rest of the body. A hemoglobin test measures "
            "the amount in blood and is often included in a complete blood count. "
            "Low or high results can have many possible explanations. A result must "
            "be interpreted using the laboratory's reference interval, symptoms, "
            "medical history and other blood tests."
        ),
    },
    {
        "id": "doc_wbc",
        "title": "White Blood Cell Count",
        "topic": "Immune system and blood cells",
        "source": "MedlinePlus",
        "url": "https://medlineplus.gov/lab-tests/white-blood-count-wbc/",
        "text": (
            "White blood cells (WBCs), also called leukocytes, are part of the immune "
            "system. They travel through the blood and tissues to help the body fight "
            "infections and other diseases. The five major types are neutrophils, "
            "lymphocytes, monocytes, eosinophils and basophils. A WBC count measures "
            "the total number of white blood cells; a differential measures the "
            "different types. High or low results have several possible causes and "
            "must be interpreted in clinical context."
        ),
    },
    {
        "id": "doc_platelets",
        "title": "Platelets and Blood Clotting",
        "topic": "Platelet count",
        "source": "MedlinePlus",
        "url": "https://medlineplus.gov/lab-tests/complete-blood-count-cbc/",
        "text": (
            "Platelets are small blood components that help stop bleeding by forming "
            "clots. A complete blood count includes a platelet count. A platelet "
            "result outside a laboratory's reference interval can have different "
            "causes; a single count is not enough to diagnose a condition."
        ),
    },
    {
        "id": "doc_glucose",
        "title": "Blood Glucose Test",
        "topic": "Blood sugar",
        "source": "MedlinePlus",
        "url": "https://medlineplus.gov/lab-tests/blood-glucose-test/",
        "text": (
            "Blood glucose is the amount of sugar (glucose) in the blood. A fasting "
            "blood glucose test measures glucose after at least eight hours without "
            "food or drink other than water. Glucose tests may be used to screen for "
            "or monitor prediabetes and diabetes. Different glucose tests have "
            "different preparation and interpretation rules. Results should be "
            "interpreted by a healthcare professional; one result alone may not "
            "establish a diagnosis."
        ),
    },
    {
        "id": "doc_cholesterol",
        "title": "Cholesterol Levels",
        "topic": "Cholesterol and cardiovascular health",
        "source": "MedlinePlus",
        "url": "https://medlineplus.gov/lab-tests/cholesterol-levels/",
        "text": (
            "A cholesterol test measures fats in the blood, including total "
            "cholesterol and commonly LDL and HDL cholesterol. LDL is often called "
            "bad cholesterol and HDL good cholesterol because their relationships "
            "to cardiovascular risk differ. Interpretation depends on which "
            "cholesterol measure is tested and the person's age and overall risk. "
            "A total cholesterol number alone does not provide a complete assessment."
        ),
    },
    {
        "id": "doc_creatinine",
        "title": "Creatinine Test",
        "topic": "Kidney function",
        "source": "MedlinePlus",
        "url": "https://medlineplus.gov/lab-tests/creatinine-test/",
        "text": (
            "Creatinine is a waste product made during normal muscle activity. "
            "Healthy kidneys filter creatinine from the blood and remove it in urine. "
            "Blood creatinine is used to help assess kidney health, but the level is "
            "also affected by muscle mass, diet, age and activity. Creatinine is "
            "often considered with estimated glomerular filtration rate (eGFR), "
            "urine tests and other clinical information."
        ),
    },
    {
        "id": "doc_bun",
        "title": "Blood Urea Nitrogen (BUN)",
        "topic": "Kidney function",
        "source": "MedlinePlus",
        "url": "https://medlineplus.gov/lab-tests/bun-blood-urea-nitrogen/",
        "text": (
            "A blood urea nitrogen (BUN) test measures urea nitrogen in the blood. "
            "Urea nitrogen is produced when the body breaks down proteins and is "
            "removed by the kidneys in urine. BUN is often assessed with other "
            "kidney tests. Higher or lower results can have different explanations "
            "and should not be interpreted in isolation."
        ),
    },
    {
        "id": "doc_alt",
        "title": "ALT Blood Test",
        "topic": "Liver enzymes",
        "source": "MedlinePlus",
        "url": "https://medlineplus.gov/lab-tests/alt-blood-test/",
        "text": (
            "ALT (alanine transaminase), also called SGPT, is an enzyme found mainly "
            "in the liver. When liver cells are damaged, ALT can be released into "
            "the bloodstream. An ALT test is commonly considered with other liver "
            "function tests. A high ALT result can have several possible causes and "
            "does not by itself show how much liver damage is present or establish "
            "a diagnosis."
        ),
    },
    {
        "id": "doc_vitamin_d",
        "title": "Vitamin D Test",
        "topic": "Vitamin D and bone health",
        "source": "MedlinePlus",
        "url": "https://medlineplus.gov/lab-tests/vitamin-d-test/",
        "text": (
            "Vitamin D helps the body absorb calcium and supports healthy bones and "
            "teeth. A vitamin D test measures vitamin D in the blood. Many tests "
            "measure 25-hydroxyvitamin D (25(OH)D). Very low vitamin D can contribute "
            "to bone disorders. Test interpretation depends on the laboratory and "
            "clinical context; routine testing is not recommended for everyone."
        ),
    },
    {
        "id": "doc_osteoporosis",
        "title": "Osteoporosis",
        "topic": "Bone strength and fractures",
        "source": "National Institute of Arthritis and Musculoskeletal and Skin Diseases (NIAMS)",
        "url": "https://www.niams.nih.gov/health-topics/osteoporosis",
        "text": (
            "Osteoporosis is a bone disease in which bone mineral density and bone "
            "mass decrease, or bone structure and strength change. This can weaken "
            "bones and increase the risk of fractures. It is often called a silent "
            "disease because there may be no symptoms until a bone breaks. Risk is "
            "influenced by age, hormones, family history, certain medicines, diet "
            "and lifestyle. A healthcare professional evaluates risk and tests."
        ),
    },
    {
        "id": "doc_bmd",
        "title": "Bone Mineral Density Tests: What the Numbers Mean",
        "topic": "DEXA or DXA bone density test",
        "source": "National Institute of Arthritis and Musculoskeletal and Skin Diseases (NIAMS)",
        "url": "https://www.niams.nih.gov/health-topics/bone-mineral-density-tests-what-numbers-mean",
        "text": (
            "A bone mineral density (BMD) test measures calcium and other minerals "
            "in bone. Bones containing more minerals are denser and tend to be "
            "stronger and less likely to break. A central DXA or DEXA scan commonly "
            "measures bone density at the hip and spine. BMD testing can help identify "
            "osteoporosis, assess fracture risk and monitor treatment. A clinician "
            "interprets results using age and other relevant factors."
        ),
    },
    {
        "id": "doc_calcium_vitd_bones",
        "title": "Calcium and Vitamin D: Important for Bone Health",
        "topic": "Calcium, vitamin D and bone strength",
        "source": "National Institute of Arthritis and Musculoskeletal and Skin Diseases (NIAMS)",
        "url": "https://www.niams.nih.gov/health-topics/calcium-and-vitamin-d-important-bone-health",
        "text": (
            "Calcium is a major mineral in bones. Calcium salts help harden and "
            "strengthen bone. The body cannot make calcium, so it must come from "
            "food and, when appropriate, supplements. If dietary calcium is too low, "
            "the body can take calcium from bones over time, which may weaken them. "
            "Vitamin D helps the body absorb calcium and supports bone mineralization. "
            "Individual calcium and vitamin D needs vary by age and other factors."
        ),
    },
]

def initialize_knowledge_base() -> None:
    """Upsert the curated source set once per running app process."""
    global _KNOWLEDGE_BASE_READY
    if _KNOWLEDGE_BASE_READY:
        return

    # Remove legacy seed documents so stale keyword-only passages cannot
    # compete with the reviewed source collection after an app upgrade.
    existing = collection.get()
    existing_ids = existing.get("ids", []) if isinstance(existing, dict) else []
    if existing_ids:
        collection.delete(ids=existing_ids)

    collection.upsert(
        ids=[item["id"] for item in MEDICAL_DOCUMENTS],
        documents=[item["text"] for item in MEDICAL_DOCUMENTS],
        metadatas=[
            {
                "title": item["title"],
                "topic": item["topic"],
                "source": item["source"],
                "url": item["url"],
            }
            for item in MEDICAL_DOCUMENTS
        ],
    )
    _KNOWLEDGE_BASE_READY = True


def _language_details(language: str) -> tuple[str, str]:
    code = str(language or "en").strip().lower()
    if code in {"hindi", "hi"}:
        return "hi", "Hindi"
    if code in {"kannada", "kn"}:
        return "kn", "Kannada"
    return "en", "English"


def _fallback_answer(
    retrieved_documents: list[str],
    retrieved_metadata: list[dict[str, Any]],
    target_code: str,
    reason: str,
) -> str:
    """Show retrieved source summaries honestly when Gemini cannot generate."""
    notices = {
        "en": (
            "Gemini could not generate an answer right now. The following is "
            "reference material retrieved from the medical knowledge base, not "
            "an AI-generated answer:"
        ),
        "hi": (
            "Gemini अभी उत्तर तैयार नहीं कर सका। नीचे चिकित्सा ज्ञानकोष से प्राप्त "
            "संदर्भ सामग्री दी गई है; यह AI द्वारा तैयार किया गया उत्तर नहीं है:"
        ),
        "kn": (
            "Gemini ಈಗ ಉತ್ತರವನ್ನು ರಚಿಸಲು ಸಾಧ್ಯವಾಗಲಿಲ್ಲ. ಕೆಳಗೆ ವೈದ್ಯಕೀಯ ಜ್ಞಾನಕೋಶದಿಂದ "
            "ಪಡೆಯಲಾದ ಉಲ್ಲೇಖ ಮಾಹಿತಿಯಿದೆ; ಇದು AI ರಚಿಸಿದ ಉತ್ತರ ಅಲ್ಲ:"
        ),
    }
    parts = [notices[target_code]]
    for index, (document, metadata) in enumerate(zip(retrieved_documents, retrieved_metadata), start=1):
        title = metadata.get("title") or metadata.get("topic") or f"Source {index}"
        parts.append(f"[{index}] {title}: {document[:900]}")
    parts.append(
        "This information is for health education only, not a diagnosis or a "
        "substitute for advice from a qualified healthcare professional."
    )
    return "\n\n".join(parts)


def answer_question(query: str, language: str = "en") -> dict[str, Any]:
    """Retrieve up to three relevant sources and generate a grounded answer."""
    if not isinstance(query, str) or not query.strip():
        return {
            "status": "error",
            "query": query,
            "answer": "Please provide a valid question.",
            "sources": [],
            "insufficient_evidence": True,
            "ai_generated": False,
            "answer_mode": "error",
        }

    initialize_knowledge_base()
    target_code, target_name = _language_details(language)

    try:
        n_results = min(MAX_RETRIEVED_SOURCES, max(1, collection.count()))
        search_results = collection.query(
            query_texts=[query.strip()],
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )
    except Exception as exc:
        print(f"Medical Q&A retrieval failed: {type(exc).__name__}", flush=True)
        return {
            "status": "error",
            "query": query,
            "answer": "The medical reference search is temporarily unavailable. Please try again.",
            "sources": [],
            "insufficient_evidence": True,
            "ai_generated": False,
            "answer_mode": "error",
        }

    documents = (search_results.get("documents") or [[]])[0] or []
    metadata_list = (search_results.get("metadatas") or [[]])[0] or []
    distances = (search_results.get("distances") or [[]])[0] or []

    if not documents or (distances and float(distances[0]) > MAX_DISTANCE):
        rejection = {
            "en": "I could not find sufficiently relevant, verified material in the current medical knowledge base for this question. Please consult a qualified healthcare professional.",
            "hi": "इस प्रश्न के लिए वर्तमान चिकित्सा ज्ञानकोष में पर्याप्त प्रासंगिक और सत्यापित जानकारी नहीं मिली। कृपया किसी योग्य स्वास्थ्य विशेषज्ञ से सलाह लें।",
            "kn": "ಈ ಪ್ರಶ್ನೆಗೆ ಪ್ರಸ್ತುತ ವೈದ್ಯಕೀಯ ಜ್ಞಾನಕೋಶದಲ್ಲಿ ಸಾಕಷ್ಟು ಸಂಬಂಧಿತ ಮತ್ತು ದೃಢೀಕೃತ ಮಾಹಿತಿ ಸಿಗಲಿಲ್ಲ. ದಯವಿಟ್ಟು ಅರ್ಹ ಆರೋಗ್ಯ ವೃತ್ತಿಪರರನ್ನು ಸಂಪರ್ಕಿಸಿ.",
        }
        return {
            "status": "success",
            "query": query,
            "answer": rejection[target_code],
            "sources": [],
            "insufficient_evidence": True,
            "ai_generated": False,
            "answer_mode": "insufficient_evidence",
            "safety_message": "MediLens AI supports health awareness and does not replace professional medical advice.",
        }

    # Keep the best match and any additional results within the same relevance
    # distance threshold. Do not send obviously unrelated references to Gemini.
    selected_documents: list[str] = []
    selected_metadata: list[dict[str, Any]] = []
    for index, document in enumerate(documents):
        distance = float(distances[index]) if index < len(distances) else 0.0
        if index == 0 or distance <= MAX_DISTANCE:
            if isinstance(document, str) and document.strip():
                selected_documents.append(document)
                selected_metadata.append(
                    metadata_list[index] if index < len(metadata_list) and metadata_list[index] else {}
                )

    sources = [
        {
            "source": metadata.get("source", "Medical reference"),
            "title": metadata.get("title", metadata.get("topic", f"Source {index}")),
            "topic": metadata.get("topic", "Retrieved medical reference"),
            "url": metadata.get("url"),
            "excerpt": document[:220] + ("..." if len(document) > 220 else ""),
        }
        for index, (document, metadata) in enumerate(
            zip(selected_documents, selected_metadata), start=1
        )
    ]

    source_context = "\n\n".join(
        f"[{index}] {metadata.get('title', metadata.get('topic', 'Medical reference'))}\n"
        f"Source: {metadata.get('source', 'Medical reference')}\n"
        f"Content: {document}"
        for index, (document, metadata) in enumerate(
            zip(selected_documents, selected_metadata), start=1
        )
    )

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        answer = _fallback_answer(
            selected_documents, selected_metadata, target_code, "API key not configured"
        )
        return {
            "status": "success",
            "query": query,
            "answer": answer,
            "sources": sources,
            "insufficient_evidence": False,
            "ai_generated": False,
            "answer_mode": "verified_reference",
            "safety_message": "This is retrieved reference material, not an AI-generated answer or a diagnosis.",
        }

    system_instruction = (
        "You are MediLens AI, an evidence-grounded medical education assistant. "
        "Answer the user's exact question using only the supplied numbered medical "
        "reference passages. Do not introduce medical facts, reference ranges, "
        "diagnoses, causes, medication advice or treatment plans that are not "
        "supported by those passages. If the passages do not answer the question, "
        "say that the available references are insufficient. Explain uncertainty "
        "and remind the user that a clinician interprets personal results. "
        f"Write the answer in {target_name}. Cite factual statements using the "
        "provided source numbers, for example [1] or [2]. Never invent citations."
    )
    prompt = (
        f"User question:\n{query.strip()}\n\n"
        f"Retrieved medical references:\n{source_context}\n\n"
        "Write a clear, direct answer to the question. Use only the references above. "
        "Use short paragraphs or a few concise bullets when helpful. Cite the "
        "relevant source numbers. If the evidence is incomplete, state that clearly."
    )

    client = None
    model_errors: list[str] = []
    try:
        client = genai.Client(
            api_key=api_key.strip().strip("'\""),
            http_options=types.HttpOptions(timeout=REQUEST_TIMEOUT_MS),
        )

        answer_text = None
        for model_name in (PRIMARY_MODEL, FALLBACK_MODEL):
            try:
                print(f"Medical Q&A using {model_name}", flush=True)
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=system_instruction,
                        max_output_tokens=400,
                    ),
                )
                if response and response.text and response.text.strip():
                    answer_text = response.text.strip()
                    break
                model_errors.append(f"{model_name}: empty response")
            except Exception as exc:
                model_errors.append(f"{model_name}: {type(exc).__name__}")
                print(
                    f"Medical Q&A {model_name} failed: {type(exc).__name__}",
                    flush=True,
                )

        if answer_text:
            return {
                "status": "success",
                "query": query,
                "answer": answer_text,
                "sources": sources,
                "insufficient_evidence": False,
                "ai_generated": True,
                "answer_mode": "gemini",
                "model": model_name,
                "safety_message": "MediLens AI supports health awareness and does not replace professional medical advice.",
            }

        print("Medical Q&A falling back to source text: " + "; ".join(model_errors), flush=True)
        answer = _fallback_answer(
            selected_documents, selected_metadata, target_code, "; ".join(model_errors)
        )
        return {
            "status": "success",
            "query": query,
            "answer": answer,
            "sources": sources,
            "insufficient_evidence": False,
            "ai_generated": False,
            "answer_mode": "verified_reference",
            "safety_message": "This is retrieved reference material, not an AI-generated answer or a diagnosis.",
        }

    except Exception as exc:
        print(f"Medical Q&A client error: {type(exc).__name__}", flush=True)
        answer = _fallback_answer(
            selected_documents, selected_metadata, target_code, type(exc).__name__
        )
        return {
            "status": "success",
            "query": query,
            "answer": answer,
            "sources": sources,
            "insufficient_evidence": False,
            "ai_generated": False,
            "answer_mode": "verified_reference",
            "safety_message": "This is retrieved reference material, not an AI-generated answer or a diagnosis.",
        }
    finally:
        if client is not None:
            client.close()


if __name__ == "__main__":
    result = answer_question(
        "How do calcium and vitamin D work together to maintain strong bones?",
        language="en",
    )
    print(result)
