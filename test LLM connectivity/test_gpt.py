from openai import OpenAI
from dotenv import load_dotenv
import os


def test_openai_connection():
    # Load environment variables
    load_dotenv()
    
    # Configure OpenAI client
    api_key = os.getenv('OPENAI_API_KEY')
    
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in .env file")
    
    client = OpenAI(
        api_key=api_key,
    )

    try:
        # Create a simple prompt and generate a response
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",  # Using a standard model
            messages=[
                {"role": "user", "content": "write me a haiku about a cat!"}
            ]
        )
        print("OpenAI Response:", response.choices[0].message.content)
        print("\nConnection test successful! ✅")
        return True
    except Exception as e:
        print(f"Connection test failed! ❌\nError: {str(e)}")
        return False

if __name__ == "__main__":
    test_openai_connection()