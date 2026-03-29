from dotenv import load_dotenv
load_dotenv()

import os, time
from openai import OpenAI

client = OpenAI(
    base_url=os.environ.get("OPENAI_BASE_URL"),
    api_key=os.environ.get("OPENAI_API_KEY")
)

print(f"Testing with base URL: {os.environ.get('OPENAI_BASE_URL')}")
print()

models_to_try = [
    "llama-3.3-70b-versatile",
    "llama-3.1-70b-versatile",
    "mixtral-8x7b-32768",
    "llama-3.1-8b-instant",
]

for model in models_to_try:
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Say exactly: working"}],
            max_tokens=10
        )
        print(f"✓ {model}: {response.choices[0].message.content.strip()}")
    except Exception as e:
        print(f"✗ {model}: {str(e)[:80]}")
    time.sleep(2)