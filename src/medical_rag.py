import os
import chromadb
from dotenv import load_dotenv
from google import genai
from google.genai import types

# 1. Load the API key. The modern Google Gen AI SDK is used below.
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if api_key:
    api_key = api_key.strip().strip("'\"")

# 2. Connect to local persistent Chroma DB
chroma_client = chromadb.PersistentClient(path="./chroma_data")
collection = chroma_client.get_or_create_collection(name="medical_knowledge")

# Multilingual verified references (guarantees proper language output even on API quota limits)
FALLBACK_KNOWLEDGE = {
    "doc_glucose": {
        "en": "Fasting blood glucose normal range is 70 to 99 mg/dL. Values between 100 and 125 mg/dL indicate prediabetes. Levels of 126 mg/dL or higher indicate diabetes.",
        "hi": "फास्टिंग ब्लड ग्लूकोज का सामान्य स्तर 70 से 99 mg/dL होता है। 100 से 125 mg/dL का स्तर प्रीडायबिटीज और 126 mg/dL या उससे अधिक का स्तर मधुमेह (डायबिटीज) का संकेत देता है।",
        "kn": "ಉಪವಾಸದ ರಕ್ತದ ಗ್ಲೂಕೋಸ್‌ನ ಸಾಮಾನ್ಯ ಪ್ರಮಾಣವು 70 ರಿಂದ 99 mg/dL ಆಗಿದೆ. 100 ರಿಂದ 125 mg/dL ನಡುವಿನ ಮೌಲ್ಯಗಳು ಪೂರ್ವ ಮಧುಮೇಹವನ್ನು ಮತ್ತು 126 mg/dL ಅಥವಾ ಅದಕ್ಕಿಂತ ಹೆಚ್ಚಿನ ಮಟ್ಟವು ಮಧುಮೇಹವನ್ನು ಸೂಚಿಸುತ್ತದೆ.",
    },
    "doc_hemoglobin": {
        "en": "Standard hemoglobin reference ranges are 13.8 to 17.2 g/dL for adult men and 12.1 to 15.1 g/dL for adult women. Low levels indicate anemia.",
        "hi": "हीमोग्लोबिन का सामान्य स्तर पुरुषों के लिए 13.8 से 17.2 g/dL और महिलाओं के लिए 12.1 से 15.1 g/dL है। कम स्तर एनीमिया (खून की कमी) का संकेत है।",
        "kn": "ಹಿಮೋಗ್ಲೋಬಿನ್ ಸಾಮಾನ್ಯ ಶ್ರೇಣಿ ಪುರುಷರಿಗೆ 13.8-17.2 g/dL ಮತ್ತು ಮಹಿಳೆಯರಿಗೆ 12.1-15.1 g/dL. ಕಡಿಮೆ ಮಟ್ಟವು ರಕ್ತಹೀನತೆಯನ್ನು ಸೂಚಿಸುತ್ತದೆ.",
    },
    "doc_creatinine": {
        "en": "Typical adult serum creatinine ranges are 0.7 to 1.3 mg/dL for men and 0.6 to 1.1 mg/dL for women. Elevated levels suggest impaired kidney filtration.",
        "hi": "सीरम क्रिएटिनिन का सामान्य स्तर पुरुषों में 0.7 से 1.3 mg/dL और महिलाओं में 0.6 से 1.1 mg/dL होता है। बढ़ा हुआ स्तर गुर्दे (किडनी) की समस्या का संकेत दे सकता है।",
        "kn": "ಸೀರಮ್ ಕ್ರಿಯೇಟಿನೈನ್ ಸಾಮಾನ್ಯ ಮಟ್ಟವು ಪುರುಷರಲ್ಲಿ 0.7 ರಿಂದ 1.3 mg/dL ಮತ್ತು ಮಹಿಳೆಯರಲ್ಲಿ 0.6 ರಿಂದ 1.1 mg/dL ಆಗಿದೆ. ಹೆಚ್ಚಿದ ಮಟ್ಟವು ಮೂತ್ರಪಿಂಡದ ಸಮಸ್ಯೆಯನ್ನು ಸೂಚಿಸುತ್ತದೆ.",
    },
    "doc_cholesterol": {
        "en": "Desirable total cholesterol is under 200 mg/dL. Readings from 200 to 239 mg/dL are borderline high, and 240 mg/dL or greater increases cardiovascular risk.",
        "hi": "कुल कोलेस्ट्रॉल 200 mg/dL से कम होना चाहिए। 200 से 239 mg/dL बॉर्डरलाइन उच्च और 240 mg/dL या अधिक हृदय रोग के जोखिम को बढ़ा सकता है।",
        "kn": "ಒಟ್ಟು ಕೊಲೆಸ್ಟ್ರಾಲ್ 200 mg/dL ಗಿಂತ ಕಡಿಮೆಯಿರಬೇಕು. 200 ರಿಂದ 239 mg/dL ಗಡಿರೇಖೆಯಾಗಿದ್ದು, 240 mg/dL ಗಿಂತ ಹೆಚ್ಚು ಹೃದ್ರೋಗದ ಅಪಾಯವನ್ನು ಹೆಚ್ಚಿಸುತ್ತದೆ.",
    },
    "doc_wbc": {
        "en": "White blood cells (WBCs), also called leukocytes, are part of the immune system. They travel through the blood and tissues to help the body fight infections and other diseases. The five major types are neutrophils, lymphocytes, monocytes, eosinophils, and basophils.",
        "hi": "श्वेत रक्त कोशिकाएँ (WBC), जिन्हें ल्यूकोसाइट्स भी कहते हैं, प्रतिरक्षा प्रणाली का हिस्सा हैं। ये रक्त और ऊतकों के माध्यम से संक्रमण और अन्य बीमारियों से लड़ने में शरीर की मदद करती हैं। इनके पाँच मुख्य प्रकार न्यूट्रोफिल, लिम्फोसाइट, मोनोसाइट, ईोसिनोफिल और बेसोफिल हैं।",
        "kn": "ಬಿಳಿ ರಕ್ತ ಕಣಗಳು (WBC), ಲ್ಯೂಕೋಸೈಟ್‌ಗಳು ಎಂದೂ ಕರೆಯಲ್ಪಡುತ್ತವೆ, ದೇಹದ ರೋಗನಿರೋಧಕ ವ್ಯವಸ್ಥೆಯ ಭಾಗವಾಗಿವೆ. ಇವು ರಕ್ತ ಮತ್ತು ಅಂಗಾಂಶಗಳ ಮೂಲಕ ಸಂಚರಿಸಿ ಸೋಂಕುಗಳು ಮತ್ತು ಇತರ ಕಾಯಿಲೆಗಳ ವಿರುದ್ಧ ಹೋರಾಡಲು ಸಹಾಯ ಮಾಡುತ್ತವೆ. ಇವುಗಳ ಐದು ಮುಖ್ಯ ವಿಧಗಳು ನ್ಯೂಟ್ರೋಫಿಲ್‌ಗಳು, ಲಿಂಫೋಸೈಟ್‌ಗಳು, ಮೊನೋಸೈಟ್‌ಗಳು, ಇಯೋಸಿನೋಫಿಲ್‌ಗಳು ಮತ್ತು ಬೇಸೋಫಿಲ್‌ಗಳು.",
    },
    "doc_osteoporosis": {
        "en": "Osteoporosis is a disease in which bone mineral density or bone mass decreases, or bone structure changes. Bones become weak and brittle, increasing the chance of fractures. It often has no obvious symptoms until a fracture occurs.",
        "hi": "ऑस्टियोपोरोसिस में हड्डियों का खनिज घनत्व या हड्डियों का द्रव्यमान कम हो जाता है, या उनकी संरचना बदल जाती है। इससे हड्डियाँ कमजोर और भंगुर हो जाती हैं तथा फ्रैक्चर का जोखिम बढ़ता है। अक्सर फ्रैक्चर होने तक स्पष्ट लक्षण नहीं दिखते।",
        "kn": "ಆಸ್ಟಿಯೊಪೊರೋಸಿಸ್‌ನಲ್ಲಿ ಎಲುಬಿನ ಖನಿಜ ಸಾಂದ್ರತೆ ಅಥವಾ ಎಲುಬಿನ ದ್ರವ್ಯರಾಶಿ ಕಡಿಮೆಯಾಗುತ್ತದೆ, ಅಥವಾ ಎಲುಬಿನ ರಚನೆ ಬದಲಾಗುತ್ತದೆ. ಇದರಿಂದ ಎಲುಬುಗಳು ದುರ್ಬಲವಾಗಿ ಸುಲಭವಾಗಿ ಮುರಿಯುವ ಸಾಧ್ಯತೆ ಹೆಚ್ಚುತ್ತದೆ. ಅನೇಕ ಬಾರಿ ಮುರಿತವಾಗುವವರೆಗೆ ಸ್ಪಷ್ಟ ಲಕ್ಷಣಗಳು ಕಾಣಿಸುವುದಿಲ್ಲ.",
    },
    "doc_bmd": {
        "en": "A bone mineral density (BMD) test measures minerals in bone. Bones with more minerals are denser and tend to be stronger and less likely to break. A central DXA scan commonly measures bone density at the hip and spine. A clinician interprets the result in the context of age and other factors.",
        "hi": "बोन मिनरल डेंसिटी (BMD) जाँच हड्डियों में खनिजों की मात्रा मापती है। अधिक खनिज वाली हड्डियाँ अधिक घनी होती हैं और सामान्यतः मजबूत होती हैं तथा कम टूटती हैं। केंद्रीय DXA स्कैन में आमतौर पर कूल्हे और रीढ़ की हड्डी की घनता मापी जाती है। चिकित्सक उम्र और अन्य कारकों के संदर्भ में परिणाम समझते हैं।",
        "kn": "ಮೂಳೆ ಖನಿಜ ಸಾಂದ್ರತೆ (BMD) ಪರೀಕ್ಷೆಯು ಎಲುಬಿನಲ್ಲಿರುವ ಖನಿಜಗಳನ್ನು ಅಳೆಯುತ್ತದೆ. ಹೆಚ್ಚು ಖನಿಜಗಳಿರುವ ಎಲುಬುಗಳು ಹೆಚ್ಚು ಸಾಂದ್ರವಾಗಿದ್ದು ಸಾಮಾನ್ಯವಾಗಿ ಬಲವಾಗಿರುತ್ತವೆ ಮತ್ತು ಸುಲಭವಾಗಿ ಮುರಿಯುವುದಿಲ್ಲ. ಕೇಂದ್ರ DXA ಸ್ಕ್ಯಾನ್ ಸಾಮಾನ್ಯವಾಗಿ ಸೊಂಟ ಮತ್ತು ಬೆನ್ನೆಲುಬಿನ ಸಾಂದ್ರತೆಯನ್ನು ಅಳೆಯುತ್ತದೆ. ವೈದ್ಯರು ವಯಸ್ಸು ಮತ್ತು ಇತರ ಅಂಶಗಳೊಂದಿಗೆ ಫಲಿತಾಂಶವನ್ನು ಅರ್ಥೈಸುತ್ತಾರೆ.",
    },
    "doc_calcium_vitd_bones": {
        "en": "Calcium is a major mineral in bones and helps build and maintain strong bones. Vitamin D helps the body absorb calcium and supports normal bone mineralization. Getting adequate calcium and vitamin D supports bone health, but individual needs vary.",
        "hi": "कैल्शियम हड्डियों का एक प्रमुख खनिज है और मजबूत हड्डियों के निर्माण तथा रखरखाव में मदद करता है। विटामिन D शरीर को कैल्शियम अवशोषित करने में मदद करता है और हड्डियों के सामान्य खनिजीकरण को सहारा देता है। पर्याप्त कैल्शियम और विटामिन D हड्डियों के स्वास्थ्य में सहायक हैं, लेकिन प्रत्येक व्यक्ति की आवश्यकता अलग हो सकती है।",
        "kn": "ಕ್ಯಾಲ್ಸಿಯಂ ಎಲುಬಿನ ಪ್ರಮುಖ ಖನಿಜವಾಗಿದ್ದು, ಬಲವಾದ ಎಲುಬುಗಳನ್ನು ನಿರ್ಮಿಸಲು ಮತ್ತು ಕಾಪಾಡಲು ಸಹಾಯ ಮಾಡುತ್ತದೆ. ವಿಟಮಿನ್ D ದೇಹವು ಕ್ಯಾಲ್ಸಿಯಂ ಅನ್ನು ಹೀರಿಕೊಳ್ಳಲು ಮತ್ತು ಸಾಮಾನ್ಯ ಎಲುಬಿನ ಖನಿಜೀಕರಣಕ್ಕೆ ಸಹಾಯ ಮಾಡುತ್ತದೆ. ಸಾಕಷ್ಟು ಕ್ಯಾಲ್ಸಿಯಂ ಮತ್ತು ವಿಟಮಿನ್ D ಎಲುಬಿನ ಆರೋಗ್ಯಕ್ಕೆ ಬೆಂಬಲ ನೀಡುತ್ತವೆ, ಆದರೆ ಪ್ರತಿಯೊಬ್ಬರ ಅಗತ್ಯಗಳು ವಿಭಿನ್ನವಾಗಿರುತ್ತವೆ.",
    },
}


def initialize_knowledge_base():
    """Upserts the verified reference set, including new references, without duplicating IDs."""

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
        (
            "White blood cells (WBCs), also called leukocytes, are part of the immune "
            "system. They travel through the blood and tissues to help the body fight "
            "infections and other diseases. The five major types are neutrophils, "
            "lymphocytes, monocytes, eosinophils, and basophils."
        ),
        (
            "Osteoporosis is a disease in which bone mineral density or bone mass "
            "decreases, or the structure and strength of bone changes. This can make "
            "bones weak and brittle and increase the risk of fractures. Osteoporosis "
            "often has no obvious symptoms until a bone breaks. Bone mineral density "
            "testing can help identify osteoporosis and assess fracture risk."
        ),
        (
            "A bone mineral density (BMD) test measures calcium and other minerals in "
            "bone. Bones with more minerals are denser, tend to be stronger, and are "
            "less likely to break. A central dual-energy x-ray absorptiometry scan, "
            "called DXA or DEXA, commonly measures bone density at the hip and spine. "
            "A clinician interprets BMD results using the person's age and clinical context."
        ),
        (
            "Calcium is a major mineral component of bones and helps build and maintain "
            "strong bones. Vitamin D helps the body absorb calcium and supports normal "
            "bone mineralization. Adequate calcium and vitamin D support bone health, "
            "but individual needs vary by age and other factors."
        ),
    ]

    metadatas = [
        {"source": "NIH MedlinePlus", "topic": "Blood Glucose", "page": 1, "doc_id": "MP-GLU-01"},
        {"source": "CDC Laboratory Reference", "topic": "Hemoglobin", "page": 3, "doc_id": "CDC-HEM-02"},
        {"source": "WHO Clinical Guidelines", "topic": "Serum Creatinine", "page": 7, "doc_id": "WHO-CRE-03"},
        {"source": "National Heart, Lung, and Blood Institute", "topic": "Cholesterol", "page": 2, "doc_id": "NHLBI-CHOL-04"},
        {
            "source": "MedlinePlus",
            "topic": "White Blood Cells",
            "doc_id": "MP-WBC-05",
            "url": "https://medlineplus.gov/lab-tests/white-blood-count-wbc/",
        },
        {
            "source": "National Institute of Arthritis and Musculoskeletal and Skin Diseases (NIAMS)",
            "topic": "Osteoporosis",
            "doc_id": "NIAMS-OSTEO-01",
            "url": "https://www.niams.nih.gov/health-topics/osteoporosis",
        },
        {
            "source": "National Institute of Arthritis and Musculoskeletal and Skin Diseases (NIAMS)",
            "topic": "Bone Mineral Density Tests",
            "doc_id": "NIAMS-BMD-02",
            "url": "https://www.niams.nih.gov/health-topics/bone-mineral-density-tests-what-numbers-mean",
        },
        {
            "source": "National Institute of Arthritis and Musculoskeletal and Skin Diseases (NIAMS)",
            "topic": "Calcium and Vitamin D for Bone Health",
            "doc_id": "NIAMS-BONE-03",
            "url": "https://www.niams.nih.gov/health-topics/calcium-and-vitamin-d-important-bone-health",
        },
    ]

    ids = [
        "doc_glucose", "doc_hemoglobin", "doc_creatinine", "doc_cholesterol",
        "doc_wbc", "doc_osteoporosis", "doc_bmd", "doc_calcium_vitd_bones",
    ]
    collection.upsert(documents=medical_documents, metadatas=metadatas, ids=ids)


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
    initialize_knowledge_base()
    # Normalize language code to en, hi, or kn
    lang_code = str(language).strip().lower()
    if lang_code in ["hindi", "hi"]:
        target_code = "hi"
        target_name = "Hindi"
    elif lang_code in ["kannada", "kn"]:
        target_code = "kn"
        target_name = "Kannada"
    else:
        target_code = "en"
        target_name = "English"

    # 1. Query Chroma DB for top matching reference document
    search_results = collection.query(query_texts=[query], n_results=1)

    documents = search_results.get("documents", [[]])[0]
    metadatas = search_results.get("metadatas", [[]])[0]
    ids = search_results.get("ids", [[]])[0]
    distances = search_results.get("distances", [[]])[0]

    # Guardrail: Rejection threshold if no literature matches the question
    if not documents or (distances and distances[0] > 1.5):
        rejection_messages = {
            "en": "I could not find verified medical literature in the knowledge base regarding this specific topic. Please consult your physician.",
            "hi": "मुझे इस विषय से संबंधित ज्ञानकोष में सत्यापित चिकित्सा संदर्भ नहीं मिले। कृपया अपने चिकित्सक से परामर्श लें।",
            "kn": "ಈ ನಿರ್ದಿಷ್ಟ ವಿಷಯದ ಕುರಿತು ದೃಢೀಕೃತ ವೈದ್ಯಕೀಯ ಸಾಹಿತ್ಯ ಕಂಡುಬಂದಿಲ್ಲ. ದಯವಿಟ್ಟು ನಿಮ್ಮ ವೈದ್ಯರನ್ನು ಸಂಪರ್ಕಿಸಿ.",
        }
        return {
            "status": "success",
            "query": query,
            "answer": rejection_messages.get(target_code, rejection_messages["en"]),
            "sources": [],
            "insufficient_evidence": True,
            "safety_message": "MediLens AI supports health awareness and does not replace professional medical advice.",
        }

    retrieved_chunk = documents[0]
    retrieved_meta = metadatas[0]
    retrieved_id = ids[0] if ids else "doc_glucose"

    # Fetch pre-translated reference corresponding strictly to target language
    fallback_text = FALLBACK_KNOWLEDGE.get(retrieved_id, {}).get(target_code, retrieved_chunk)

    # 2. Generate an answer with Gemini when the API key is configured.
    # If the request fails, use the matching verified reference text below.
    answer_text = None
    if api_key:
        system_instruction = (
            "You are the Medical Assistant for MediLens AI.\n"
            "Answer the user's exact question using ONLY the supplied verified medical context.\n"
            "Do not switch to a different blood test or medical topic. If the context does not "
            "answer the question, say that the knowledge base does not contain enough evidence.\n"
            "Do NOT diagnose, prescribe, or recommend treatment. Explain clearly in 2-3 sentences.\n"
            f"Write the entire answer in {target_name}."
        )
        prompt = (
            f"Verified medical reference context:\n{retrieved_chunk}\n\n"
            f"User question: {query}\n\n"
            "Answer only the question asked. Do not provide unrelated reference ranges."
        )

        client = None
        try:
            client = genai.Client(
                api_key=api_key,
                http_options=types.HttpOptions(timeout=15000),
            )
            response = client.models.generate_content(
                model="gemini-3.8-flash",
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    temperature=0.1,
                    max_output_tokens=150,
                ),
            )
            if response and response.text:
                answer_text = response.text.strip()
        except Exception as exc:
            # Keep runtime logs useful without ever printing the API key.
            print(f"Medical Q&A Gemini request failed: {type(exc).__name__}", flush=True)
        finally:
            if client is not None:
                client.close()

    # If Gemini quota exceeded or key missing, format fallback in the exact selected language
    if not answer_text:
        source_name = retrieved_meta.get("source", "Medical Literature")
        if target_code == "hi":
            answer_text = f"{source_name} के सत्यापित संदर्भों के अनुसार: {fallback_text}"
        elif target_code == "kn":
            answer_text = f"{source_name} ನ ದೃಢೀಕೃತ ಉಲ್ಲೇಖಗಳ ಪ್ರಕಾರ: {fallback_text}"
        else:
            answer_text = f"Based on verified references from {source_name}: {fallback_text}"

    return {
        "status": "success",
        "query": query,
        "answer": answer_text,
        "sources": [
            {
                "source": retrieved_meta.get("source"),
                "topic": retrieved_meta.get("topic"),
                "page": retrieved_meta.get("page"),
                "url": retrieved_meta.get("url"),
                "excerpt": retrieved_chunk[:120] + "...",
            }
        ],
        "insufficient_evidence": False,
        "safety_message": "MediLens AI supports health awareness and does not replace professional medical advice.",
    }


# Standalone verification test: Tests ONLY the single chosen language
if __name__ == "__main__":
    initialize_knowledge_base()

    # Change to "en", "hi", or "kn" to test that specific language
    selected_test_language = "en"
    test_question = "Why is my fasting blood glucose 135?"

    print(f"\n--- Testing Query ({selected_test_language.upper()}): '{test_question}' ---")
    result = answer_question(test_question, language=selected_test_language)

    print("\nAnswer:\n", result["answer"])
    print("\nSources:\n", result["sources"])
    print("\nSafety Disclaimer:\n", result["safety_message"])