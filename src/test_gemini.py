import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY not found in .env")

client = genai.Client(api_key=api_key)

try:
    print("Connecting to Gemini...", flush=True)

    stream = client.interactions.create(
        model="gemini-3.8-flash",
        input="Explain hemoglobin in simple English in two sentences.",
        stream=True
    )

    print("Gemini is responding:", flush=True)

    received_text = False

    for event in stream:
        if (
            event.event_type == "step.delta"
            and hasattr(event, "delta")
            and hasattr(event.delta, "text")
        ):
            print(event.delta.text, end="", flush=True)
            received_text = True

    if not received_text:
        print("\nNo text received. Check the returned events.")

    print()

except Exception as e:
    print(f"\nGemini API error: {type(e).__name__}: {e}")

finally:
    client.close()