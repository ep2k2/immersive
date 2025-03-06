import google.generativeai as genai
from dotenv import load_dotenv
import os

def test_gemini_connection():
    # Load environment variables
    load_dotenv()
    
    # Configure Gemini
    api_key = os.getenv('GOOGLE_GEMINI_API_KEY')
    if not api_key:
        raise ValueError("GOOGLE_GEMINI_API_KEY not found in .env file")
    
    genai.configure(api_key=api_key)

    try:
        # Create a model instance and generate a simple response
        model = genai.GenerativeModel('gemini-2.0-flash')
        response = model.generate_content('Say hello!')
        print("Gemini Response:", response.text)
        print("\nConnection test successful! ✅")
        return True
    except Exception as e:
        print(f"Connection test failed! ❌\nError: {str(e)}")
        return False

if __name__ == "__main__":
    test_gemini_connection() 