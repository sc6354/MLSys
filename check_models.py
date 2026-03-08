import os
import google.generativeai as genai

# Grab the API key
api_key = os.environ.get("GOOGLE_API_KEY")
genai.configure(api_key=api_key)

print("Here are the models your API key has access to:")
print("-" * 50)

# Loop through and print every available model that supports text generation
try:
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods:
            print(m.name)
except Exception as e:
    print(f"Error connecting to API: {e}")