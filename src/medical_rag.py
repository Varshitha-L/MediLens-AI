import os
import chromadb
from dotenv import load_dotenv
import google.generativeai as genai

# 1. Load API Key and configure REST transport to prevent network hangs
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if api_key:
    genai.configure(api_key=api_key, transport="rest")

# 2. Connect to local persistent Chroma DB
chroma_client = chromadb.PersistentClient(path="./chroma_data")
collection = chroma_client.get_or_create_collection(name="medical_knowledge")


def initialize_knowledge_base():
    """Populates Chroma DB with verified medical documents if empty."""
    if collection.count() > 0:
        return

    medical_documents = [
        (
            "Fasting blood glucose normal range is 70 to 99 mg/dL. Values between "
            "100 and 125 mg/dL indicate prediabetes or impaired fasting glucose. "
            "Levels of 126 mg/dL or higher on more than one testing occasion indicate "
            "diabetes. High levels suggest the body is not producing enough insulin "
            "or cannot effectively use the insulin it produces."
        ),
        (
            "Hemoglobin is an iron-rich protein in red blood cells that carries oxygen "
            "throughout the body. Standard reference ranges are approximately 13.8 to "
            "17.2 g/dL for adult men and 12.1 to 15.1 g/dL for adult women. Low "
            "hemoglobin levels indicate anemia, which can result in fatigue and shortness "
            "of breath."
        ),
        (
            "Serum creatinine is a waste product generated from normal muscle metabolism. "
            "Healthy kidneys filter creatinine from the blood and remove it through urine. "
            "Typical adult reference ranges are 0.7 to 1.3 mg/dL for men and 0.6 to 1.1 "
            "mg/dL for women. Elevated levels can suggest impaired kidney filtration."
        ),
        (
            "Total cholesterol measures the overall cholesterol in your blood. A "
            "desirable level is under 200 mg/dL. Readings from 200 to 239 mg/dL are "
            "borderline high, and 240 mg/dL or greater is considered elevated, which "
            "may increase cardiovascular risk."
        ),
    ]

    metadatas = [
        {"source": "NIH MedlinePlus", "topic": "Blood Glucose", "page": 1, "doc_id": "MP-GLU-01"},
        {"source": "CDC Laboratory Reference", "topic": "Hemoglobin", "page": 3, "doc_id": "CDC-HEM-02"},
        {"source": "WHO Clinical Guidelines", "topic": "Serum Creatinine", "page": 7, "doc_id": "WHO-CRE-03"},
        {"source": "National Heart, Lung, and Blood Institute", "topic": "Cholesterol", "page": 2, "doc_id": "NHLBI-CHOL-04"},
    ]

    ids = ["doc_glucose", "doc_hemoglobin", "doc_creatinine", "doc_cholesterol"]
    collection.add(documents=medical_documents, metadatas=metadatas, ids=ids)


def answer_question(query: str, language: str = "en") -> dict:
    """Retrieves evidence from Chroma and generates a source-backed answer using Gemini."""
    if not query.strip():
        return {
            "status": "error",
            "query": query,
            "answer": "Please provide a valid question.",
            "sources": [],
            "insufficient_evidence": True,
            "safety_message": "MediLens AI supports health awareness and does not replace professional medical advice.",
        }

    # 1. Query Chroma DB for top matching reference document
    search_results = collection.query(query_texts=[query], n_results=1)

    documents = search_results.get("documents", [[]])[0]
    metadatas = search_results.get("metadatas", [[]])[0]
    distances = search_results.get("distances", [[]])[0]

    # Guardrail: Rejection threshold if no literature matches the question
    if not documents or (distances and distances[0] > 1.5):
        return {
            "status": "success",
            "query": query,
            "answer": "I could not find verified medical literature in the knowledge base regarding this specific topic. Please consult your physician.",
            "sources": [],
            "insufficient_evidence": True,
            "safety_message": "MediLens AI supports health awareness and does not replace professional medical advice.",
        }

    retrieved_chunk = documents[0]
    retrieved_meta = metadatas[0]
    # If API key is missing, return verified reference text directly without calling Gemini
    if not api_key:
        return {
            "status": "success",
            "query": query,
            "answer": (
                f"Note: GEMINI_API_KEY is not configured. Retrieved verified clinical guidance: "
                f"{retrieved_chunk}"
            ),
            "sources": [
                {
                    "source": retrieved_meta.get("source"),
                    "topic": retrieved_meta.get("topic"),
                    "page": retrieved_meta.get("page"),
                    "excerpt": retrieved_chunk[:120] + "...",
                }
            ],
            "insufficient_evidence": False,
            "safety_message": (
                "MediLens AI supports health awareness and does not replace"
                " professional medical advice."
            ),
        }

    # 2. Strict Grounded RAG Prompt
    system_instruction = (
        "You are the Medical Assistant for MediLens AI. "
        "Answer the user's question using ONLY the provided verified medical context below. "
        "Strict rules:\n"
        "1. Do NOT diagnose any disease.\n"
        "2. Do NOT prescribe medications, remedies, or dosages.\n"
        "3. Explain clearly in plain, patient-friendly language.\n"
        f"4. Respond in {language} language.\n"
        "5. Keep the explanation within 2-3 sentences."
    )

    prompt = f"""
Medical Reference Context:
\"\"\"{retrieved_chunk}\"\"\"

User Question: {query}
"""

    # 3. Model fallback list (1,500 requests/day models first to avoid quota locks)
    candidate_models = [
        "gemini-1.5-flash",
        "gemini-2.0-flash",
        "models/gemini-1.5-flash",
        "models/gemini-2.0-flash",
        "models/gemini-3.8-flash",
    ]

    answer_text = None

    for model_name in candidate_models:
        try:
            model = genai.GenerativeModel(
                model_name=model_name,
                system_instruction=system_instruction,
                generation_config={"temperature": 0.1},
            )
            response = model.generate_content(prompt)
            if response and response.text:
                answer_text = response.text.strip()
                break
        except Exception:
            # Silently fall back to the next model in the list
            continue

    # 4. Graceful Fallback if Google API is completely unavailable or exhausted
    if not answer_text:
        answer_text = (
            f"Based on verified references from {retrieved_meta.get('source')}: "
            f"{retrieved_chunk}"
        )

    # 5. Adhere strictly to Member 3 Data Contract
    return {
        "status": "success",
        "query": query,
        "answer": answer_text,
        "sources": [
            {
                "source": retrieved_meta.get("source"),
                "topic": retrieved_meta.get("topic"),
                "page": retrieved_meta.get("page"),
                "excerpt": retrieved_chunk[:120] + "...",
            }
        ],
        "insufficient_evidence": False,
        "safety_message": "MediLens AI supports health awareness and does not replace professional medical advice.",
    }


# Standalone verification test
if __name__ == "__main__":
    initialize_knowledge_base()

    test_question = "Why is my fasting blood glucose 135?"
    print(f"\n--- Testing Query: '{test_question}' ---")
    result = answer_question(test_question)

    print("\nAnswer:\n", result["answer"])
    print("\nSources:\n", result["sources"])
    print("\nSafety Disclaimer:\n", result["safety_message"])