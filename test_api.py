import os
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# Get the API key
api_key = os.getenv("GEMINI_API_KEY")

if api_key:
    print("✅ API key loaded successfully!")
    # Only show the first few characters for safety
    print("Key starts with:", api_key[:7], "...")
else:
    print("❌ API key not found. Check your .env file.")
