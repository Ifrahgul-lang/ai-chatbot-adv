"""Run: python check_models.py  -> prints the models your Groq key can use."""
import os

from dotenv import load_dotenv
from groq import Groq

load_dotenv()
key = (os.getenv("GROQ_API_KEY") or "").strip().strip('"').strip("'")
print("Key found:", bool(key), "| starts with:", key[:4] + "..." if key else "-")
for m in Groq(api_key=key).models.list().data:
    print(m.id)
