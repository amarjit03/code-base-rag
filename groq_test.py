# groq_test.py

import os
from groq import Groq
from dotenv import load_dotenv
import traceback

# Load environment variables from .env
load_dotenv()

def main():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError("GROQ_API_KEY is not set in the environment or .env file.")

    try:
        # Initialize Groq client
        client = Groq(api_key=api_key)

        # Define chat messages
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Explain how list comprehensions work in Python."}
        ]

        # Make the chat request
        response = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=messages,
            temperature=0.2,
            max_tokens=300
        )

        # Display the response
        print("\n=== Groq LLM Response ===\n")
        print(response.choices[0].message.content.strip())

    except Exception as e:
        print("\n[ERROR] Failed to get response from Groq:")
        traceback.print_exc()

if __name__ == "__main__":
    main()
