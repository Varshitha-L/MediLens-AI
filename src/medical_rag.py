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
        "en": "Blood glucose is the amount of sugar in the blood. A normal fasting blood glucose level is generally 70–99 mg/dL. Levels of 100–125 mg/dL may indicate prediabetes, while 126 mg/dL or higher may indicate diabetes when confirmed by appropriate testing. Consult a healthcare professional for interpretation.",
        "hi": "ब्लड ग्लूकोज का अर्थ रक्त में शुगर की मात्रा है। सामान्य फास्टिंग ब्लड ग्लूकोज आमतौर पर 70–99 mg/dL होता है। 100–125 mg/dL प्रीडायबिटीज का संकेत हो सकता है और 126 mg/dL या अधिक डायबिटीज का संकेत हो सकता है। सही जानकारी के लिए डॉक्टर से सलाह लें।",
        "kn": "ರಕ್ತದಲ್ಲಿನ ಗ್ಲೂಕೋಸ್ ಎಂದರೆ ರಕ್ತದಲ್ಲಿರುವ ಸಕ್ಕರೆಯ ಪ್ರಮಾಣ. ಸಾಮಾನ್ಯ ಉಪವಾಸದ ರಕ್ತದ ಗ್ಲೂಕೋಸ್ ಮಟ್ಟವು ಸಾಮಾನ್ಯವಾಗಿ 70–99 mg/dL ಆಗಿರುತ್ತದೆ. 100–125 mg/dL ಪ್ರೀಡಯಾಬಿಟಿಸ್ ಸೂಚಿಸಬಹುದು ಮತ್ತು 126 mg/dL ಅಥವಾ ಹೆಚ್ಚು ಮಧುಮೇಹವನ್ನು ಸೂಚಿಸಬಹುದು. ಫಲಿತಾಂಶವನ್ನು ವೈದ್ಯರೊಂದಿಗೆ ಪರಿಶೀಲಿಸಿ."
    },

    "doc_hemoglobin": {
        "en": "Hemoglobin is a protein in red blood cells that carries oxygen throughout the body. Low hemoglobin may indicate anemia. Possible symptoms include tiredness, weakness, and dizziness. Normal ranges vary according to age, sex, pregnancy status, and laboratory. Consult a healthcare professional to understand your result.",
        "hi": "हीमोग्लोबिन लाल रक्त कोशिकाओं में पाया जाने वाला प्रोटीन है, जो शरीर में ऑक्सीजन पहुँचाता है। हीमोग्लोबिन कम होने पर एनीमिया हो सकता है। इसके लक्षण थकान, कमजोरी और चक्कर आना हो सकते हैं। सामान्य स्तर उम्र, लिंग और लैब के अनुसार बदलता है।",
        "kn": "ಹಿಮೋಗ್ಲೋಬಿನ್ ಕೆಂಪು ರಕ್ತಕಣಗಳಲ್ಲಿರುವ ಪ್ರೋಟೀನ್ ಆಗಿದ್ದು, ದೇಹದಾದ್ಯಂತ ಆಮ್ಲಜನಕವನ್ನು ಸಾಗಿಸುತ್ತದೆ. ಹಿಮೋಗ್ಲೋಬಿನ್ ಕಡಿಮೆಯಾದರೆ ರಕ್ತಹೀನತೆ ಇರಬಹುದು. ದಣಿವು, ದೌರ್ಬಲ್ಯ ಮತ್ತು ತಲೆಸುತ್ತು ಲಕ್ಷಣಗಳಾಗಿರಬಹುದು. ಸಾಮಾನ್ಯ ಮಟ್ಟವು ವಯಸ್ಸು, ಲಿಂಗ ಮತ್ತು ಪ್ರಯೋಗಾಲಯದ ಪ್ರಕಾರ ಬದಲಾಗುತ್ತದೆ."
    },

    "doc_creatinine": {
        "en": "Creatinine is a waste product produced by muscles and filtered from the blood by the kidneys. A high blood creatinine level may indicate reduced kidney filtration, but other factors can also affect it. Doctors interpret creatinine along with other tests, such as estimated GFR, and the person's medical history.",
        "hi": "क्रिएटिनिन मांसपेशियों से बनने वाला अपशिष्ट पदार्थ है, जिसे किडनी रक्त से छानकर बाहर निकालती है। रक्त में क्रिएटिनिन बढ़ना किडनी की कार्यक्षमता कम होने का संकेत हो सकता है, लेकिन अन्य कारण भी हो सकते हैं। डॉक्टर इसे अन्य जाँचों के साथ समझते हैं।",
        "kn": "ಕ್ರಿಯಾಟಿನಿನ್ ಸ್ನಾಯುಗಳಿಂದ ಉತ್ಪತ್ತಿಯಾಗುವ ತ್ಯಾಜ್ಯ ವಸ್ತುವಾಗಿದ್ದು, ಮೂತ್ರಪಿಂಡಗಳು ಅದನ್ನು ರಕ್ತದಿಂದ ಶೋಧಿಸುತ್ತವೆ. ರಕ್ತದಲ್ಲಿನ ಕ್ರಿಯಾಟಿನಿನ್ ಹೆಚ್ಚಾದರೆ ಮೂತ್ರಪಿಂಡಗಳ ಶೋಧನಾ ಸಾಮರ್ಥ್ಯ ಕಡಿಮೆಯಾಗಿರಬಹುದು. ಇತರ ಕಾರಣಗಳೂ ಇರಬಹುದು. ವೈದ್ಯರು ಇದನ್ನು ಇತರ ಪರೀಕ್ಷೆಗಳೊಂದಿಗೆ ಪರಿಶೀಲಿಸುತ್ತಾರೆ."
    },

    "doc_cholesterol": {
        "en": "Cholesterol is a fatty substance in the blood that the body needs in certain amounts. High levels of LDL cholesterol can increase the risk of heart disease. A complete lipid profile includes different cholesterol measurements and triglycerides. Results should be interpreted with age and other health risk factors.",
        "hi": "कोलेस्ट्रॉल रक्त में पाया जाने वाला एक वसायुक्त पदार्थ है, जिसकी शरीर को कुछ मात्रा में आवश्यकता होती है। LDL कोलेस्ट्रॉल अधिक होने से हृदय रोग का खतरा बढ़ सकता है। पूरी लिपिड प्रोफाइल में अलग-अलग कोलेस्ट्रॉल स्तर और ट्राइग्लिसराइड शामिल होते हैं।",
        "kn": "ಕೊಲೆಸ್ಟ್ರಾಲ್ ರಕ್ತದಲ್ಲಿರುವ ಕೊಬ್ಬಿನಂತಹ ವಸ್ತುವಾಗಿದ್ದು, ದೇಹಕ್ಕೆ ನಿರ್ದಿಷ್ಟ ಪ್ರಮಾಣದಲ್ಲಿ ಅಗತ್ಯವಿದೆ. LDL ಕೊಲೆಸ್ಟ್ರಾಲ್ ಹೆಚ್ಚಾದರೆ ಹೃದಯ ಕಾಯಿಲೆಯ ಅಪಾಯ ಹೆಚ್ಚಬಹುದು. ಸಂಪೂರ್ಣ ಲಿಪಿಡ್ ಪ್ರೊಫೈಲ್‌ನಲ್ಲಿ ವಿವಿಧ ಕೊಲೆಸ್ಟ್ರಾಲ್ ಮಟ್ಟಗಳು ಮತ್ತು ಟ್ರೈಗ್ಲಿಸರೈಡ್‌ಗಳು ಸೇರಿರುತ್ತವೆ."
    },

    "doc_fever": {
        "en": "Fever is a temporary increase in body temperature, often caused by an infection. Other symptoms may include chills, sweating, headache, and body aches. Fever alone cannot identify the cause. Seek medical advice if it persists or is concerning. Get urgent medical help if severe symptoms occur.",
        "hi": "बुखार शरीर के तापमान में अस्थायी वृद्धि है, जो अक्सर संक्रमण के कारण होता है। इसके लक्षणों में ठंड लगना, पसीना आना, सिरदर्द और शरीर में दर्द शामिल हो सकते हैं। केवल बुखार से कारण का पता नहीं चलता। बुखार बना रहे या गंभीर लक्षण हों तो चिकित्सा सहायता लें।",
        "kn": "ಜ್ವರವು ದೇಹದ ಉಷ್ಣತೆಯಲ್ಲಿ ತಾತ್ಕಾಲಿಕ ಹೆಚ್ಚಳವಾಗಿದ್ದು, ಸಾಮಾನ್ಯವಾಗಿ ಸೋಂಕಿನಿಂದ ಉಂಟಾಗಬಹುದು. ಚಳಿ, ಬೆವರು, ತಲೆನೋವು ಮತ್ತು ಮೈ ನೋವು ಇದರ ಲಕ್ಷಣಗಳಾಗಿರಬಹುದು. ಜ್ವರದಿಂದ ಮಾತ್ರ ಕಾರಣವನ್ನು ತಿಳಿಯಲು ಸಾಧ್ಯವಿಲ್ಲ. ಜ್ವರ ಮುಂದುವರಿದರೆ ಅಥವಾ ಗಂಭೀರ ಲಕ್ಷಣಗಳಿದ್ದರೆ ವೈದ್ಯಕೀಯ ಸಹಾಯ ಪಡೆಯಿರಿ."
    },

    "doc_cough": {
        "en": "A cough is a reflex that helps clear the airways. It may be caused by viral infections, allergies, asthma, or other conditions. A cough can be dry or produce mucus. Seek medical advice if it persists or worsens. Breathing difficulty, chest pain, or coughing up blood requires prompt medical assessment.",
        "hi": "खाँसी एक प्राकृतिक प्रतिक्रिया है जो साँस की नलियों को साफ करने में मदद करती है। यह वायरल संक्रमण, एलर्जी, अस्थमा या अन्य कारणों से हो सकती है। खाँसी सूखी या बलगम वाली हो सकती है। खाँसी लंबे समय तक रहे या बढ़े तो डॉक्टर से सलाह लें। साँस लेने में परेशानी, सीने में दर्द या खून वाली खाँसी में तुरंत चिकित्सा सहायता लें।",
        "kn": "ಕೆಮ್ಮು ಉಸಿರಾಟದ ಮಾರ್ಗಗಳನ್ನು ಸ್ವಚ್ಛಗೊಳಿಸಲು ಸಹಾಯ ಮಾಡುವ ಸಹಜ ಪ್ರತಿಕ್ರಿಯೆಯಾಗಿದೆ. ಇದು ವೈರಲ್ ಸೋಂಕು, ಅಲರ್ಜಿ, ಆಸ್ತಮಾ ಅಥವಾ ಇತರ ಕಾರಣಗಳಿಂದ ಉಂಟಾಗಬಹುದು. ಕೆಮ್ಮು ಒಣಗಿರಬಹುದು ಅಥವಾ ಕಫದೊಂದಿಗೆ ಇರಬಹುದು. ಕೆಮ್ಮು ಮುಂದುವರಿದರೆ ವೈದ್ಯರನ್ನು ಸಂಪರ್ಕಿಸಿ. ಉಸಿರಾಟದ ತೊಂದರೆ, ಎದೆನೋವು ಅಥವಾ ರಕ್ತದೊಂದಿಗೆ ಕೆಮ್ಮು ಬಂದರೆ ತಕ್ಷಣ ವೈದ್ಯಕೀಯ ಸಹಾಯ ಪಡೆಯಿರಿ."
    },

    "doc_sore_throat": {
        "en": "A sore throat is pain, irritation, or scratchiness in the throat. Viral infections are a common cause, but bacterial infections such as strep throat can also cause it. Not every sore throat needs antibiotics. A healthcare professional can decide whether testing or treatment is needed.",
        "hi": "गले में खराश का अर्थ गले में दर्द, जलन या खुरदरापन महसूस होना है। इसका सामान्य कारण वायरल संक्रमण है, लेकिन बैक्टीरियल संक्रमण भी इसका कारण हो सकता है। हर गले की खराश के लिए एंटीबायोटिक की आवश्यकता नहीं होती। डॉक्टर जाँच और उपचार की आवश्यकता तय कर सकते हैं।",
        "kn": "ಗಂಟಲು ನೋವು ಎಂದರೆ ಗಂಟಲಿನಲ್ಲಿ ನೋವು, ಉರಿ ಅಥವಾ ಕೆರೆತ ಉಂಟಾಗುವುದು. ವೈರಲ್ ಸೋಂಕು ಸಾಮಾನ್ಯ ಕಾರಣವಾಗಿದ್ದು, ಬ್ಯಾಕ್ಟೀರಿಯಾದ ಸೋಂಕೂ ಕಾರಣವಾಗಬಹುದು. ಪ್ರತಿಯೊಂದು ಗಂಟಲು ನೋವಿಗೂ ಆಂಟಿಬಯೋಟಿಕ್ ಅಗತ್ಯವಿಲ್ಲ. ಪರೀಕ್ಷೆ ಅಥವಾ ಚಿಕಿತ್ಸೆ ಬೇಕೇ ಎಂಬುದನ್ನು ವೈದ್ಯರು ನಿರ್ಧರಿಸುತ್ತಾರೆ."
    },

    "doc_diabetes": {
        "en": "Diabetes is a condition in which blood glucose levels become too high. It occurs when the body does not make enough insulin or cannot use insulin properly. Possible symptoms include increased thirst, frequent urination, tiredness, and unexplained weight changes. Diagnosis requires appropriate blood tests and medical evaluation.",
        "hi": "डायबिटीज एक ऐसी स्थिति है जिसमें रक्त में शुगर का स्तर अधिक हो जाता है। यह तब होता है जब शरीर पर्याप्त इंसुलिन नहीं बनाता या उसका सही उपयोग नहीं कर पाता। इसके लक्षणों में अधिक प्यास लगना, बार-बार पेशाब आना और थकान शामिल हो सकते हैं। इसकी पुष्टि के लिए रक्त जाँच और डॉक्टर की सलाह आवश्यक है।",
        "kn": "ಮಧುಮೇಹವು ರಕ್ತದಲ್ಲಿನ ಸಕ್ಕರೆಯ ಮಟ್ಟ ಹೆಚ್ಚಾಗುವ ಸ್ಥಿತಿಯಾಗಿದೆ. ದೇಹವು ಸಾಕಷ್ಟು ಇನ್ಸುಲಿನ್ ಉತ್ಪಾದಿಸದಿದ್ದಾಗ ಅಥವಾ ಅದನ್ನು ಸರಿಯಾಗಿ ಬಳಸಲಾಗದಿದ್ದಾಗ ಇದು ಸಂಭವಿಸುತ್ತದೆ. ಹೆಚ್ಚು ಬಾಯಾರಿಕೆ, ಆಗಾಗ್ಗೆ ಮೂತ್ರ ವಿಸರ್ಜನೆ ಮತ್ತು ದಣಿವು ಲಕ್ಷಣಗಳಾಗಿರಬಹುದು. ರಕ್ತ ಪರೀಕ್ಷೆ ಮತ್ತು ವೈದ್ಯಕೀಯ ಮೌಲ್ಯಮಾಪನದ ಮೂಲಕ ಇದನ್ನು ಪತ್ತೆಹಚ್ಚಲಾಗುತ್ತದೆ."
    },

    "doc_blood_pressure": {
        "en": "Blood pressure is the force of blood pushing against artery walls. It is recorded using two numbers: systolic pressure (top number) and diastolic pressure (bottom number). Persistently high blood pressure can increase the risk of heart and kidney problems. One reading alone may not establish a diagnosis; repeated measurements and medical advice are important.",
        "hi": "ब्लड प्रेशर वह दबाव है जिससे रक्त धमनियों की दीवारों पर बल डालता है। इसे दो संख्याओं में मापा जाता है: सिस्टोलिक (ऊपरी संख्या) और डायस्टोलिक (निचली संख्या)। लगातार उच्च रक्तचाप से हृदय और किडनी की समस्याओं का खतरा बढ़ सकता है। केवल एक रीडिंग से निदान नहीं किया जाता।",
        "kn": "ರಕ್ತದೊತ್ತಡ ಎಂದರೆ ರಕ್ತವು ಧಮನಿಗಳ ಗೋಡೆಗಳ ಮೇಲೆ ಬೀರುವ ಒತ್ತಡ. ಇದನ್ನು ಎರಡು ಸಂಖ್ಯೆಗಳ ಮೂಲಕ ಅಳೆಯಲಾಗುತ್ತದೆ: ಸಿಸ್ಟೋಲಿಕ್ (ಮೇಲಿನ ಸಂಖ್ಯೆ) ಮತ್ತು ಡಯಾಸ್ಟೋಲಿಕ್ (ಕೆಳಗಿನ ಸಂಖ್ಯೆ). ನಿರಂತರವಾಗಿ ಅಧಿಕ ರಕ್ತದೊತ್ತಡ ಇದ್ದರೆ ಹೃದಯ ಮತ್ತು ಮೂತ್ರಪಿಂಡದ ಸಮಸ್ಯೆಗಳ ಅಪಾಯ ಹೆಚ್ಚಬಹುದು. ಒಂದೇ ಅಳತೆಯಿಂದ ರೋಗನಿರ್ಣಯ ಮಾಡಲಾಗುವುದಿಲ್ಲ."
    }
}


# 4. Medical reference documents

MEDICAL_DOCUMENTS = [
    # Existing document 1: Glucose
    (
        "Fasting blood glucose is commonly considered normal at 70–99 mg/dL. "
        "Values between 100 and 125 mg/dL may indicate prediabetes. A fasting "
        "result of 126 mg/dL or higher can indicate diabetes when confirmed "
        "using appropriate testing."
    ),

    # Existing document 2: Hemoglobin
    (
        "Hemoglobin is a protein in red blood cells that carries oxygen. Adult "
        "reference ranges vary by laboratory and individual factors. Low "
        "hemoglobin may indicate anemia and requires clinical interpretation."
    ),

    # Existing document 3: Creatinine
    (
        "Creatinine is a waste product from muscle metabolism that is filtered "
        "by the kidneys. Elevated levels may indicate impaired kidney filtration "
        "but must be interpreted with other clinical findings."
    ),

    # Existing document 4: Cholesterol
    (
        "Total cholesterol measures cholesterol in the blood. Results should be "
        "interpreted with the complete lipid profile and the person's other "
        "cardiovascular risk factors."
    ),

    # New document 5: Fever
    (
        "Fever is a temporary increase in body temperature, often associated "
        "with an infection. Common causes include viral and bacterial infections. "
        "Other symptoms may include chills, sweating, headache and body aches. "
        "The cause of fever cannot be identified from temperature alone. "
        "Seek medical advice for persistent or concerning fever. "
        "Urgent medical assessment is important for severe symptoms."
    ),

    # New document 6: Cough
    (
        "A cough is a reflex that helps clear the airways. It may occur due to "
        "viral infections, allergies, asthma or other respiratory conditions. "
        "A cough can be dry or produce mucus. A cough with difficulty breathing, "
        "chest pain or blood requires prompt medical assessment."
    ),

    # New document 7: Sore throat
    (
        "A sore throat is pain, irritation or scratchiness in the throat. "
        "It is commonly caused by viral infections and can sometimes be caused "
        "by bacterial infections such as strep throat. Not every sore throat "
        "requires antibiotics. A healthcare professional can determine whether "
        "testing or treatment is needed."
    ),

    # New document 8: Diabetes
    (
        "Diabetes is a condition in which blood glucose levels are too high. "
        "It can occur because the body does not produce enough insulin or "
        "cannot use insulin effectively. Common symptoms may include increased "
        "thirst, frequent urination, tiredness and unexplained weight loss. "
        "Diagnosis requires appropriate blood tests and medical evaluation."
    ),

    # New document 9: Blood pressure
    (
        "Blood pressure measures the force of blood against artery walls. "
        "It is recorded using two numbers: systolic pressure over diastolic "
        "pressure. Persistently high blood pressure can increase the risk of "
        "heart and kidney problems. A single reading does not always establish "
        "a diagnosis; readings should be interpreted by a healthcare professional."
    ),
]

# These labels are existing project metadata, not independently verified citations.
MEDICAL_METADATA = [
    {"source": "NIH MedlinePlus", "topic": "Blood Glucose", "page": 1, "doc_id": "MP-GLU-01"},
    {"source": "CDC Laboratory Reference", "topic": "Hemoglobin", "page": 3, "doc_id": "CDC-HEM-02"},
    {"source": "WHO Clinical Guidelines", "topic": "Serum Creatinine", "page": 7, "doc_id": "WHO-CRE-03"},
    {"source": "NHLBI", "topic": "Cholesterol", "page": 2, "doc_id": "NHLBI-CHOL-04"},

    {"source": "Medical Knowledge Base", "topic": "Fever", "doc_id": "MED-FEV-05"},
    {"source": "Medical Knowledge Base", "topic": "Cough", "doc_id": "MED-COU-06"},
    {"source": "Medical Knowledge Base", "topic": "Sore Throat", "doc_id": "MED-THR-07"},
    {"source": "Medical Knowledge Base", "topic": "Diabetes", "doc_id": "MED-DIA-08"},
    {"source": "Medical Knowledge Base", "topic": "Blood Pressure", "doc_id": "MED-BP-09"},
]

DOCUMENT_IDS = [
    "doc_glucose",
    "doc_hemoglobin",
    "doc_creatinine",
    "doc_cholesterol",
    "doc_fever",
    "doc_cough",
    "doc_sore_throat",
    "doc_diabetes",
    "doc_blood_pressure",
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