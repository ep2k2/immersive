from together import Together
from dotenv import load_dotenv
import os

def test_llama_vision_connection():
    # Load environment variables
    load_dotenv()
    
    # Access the API key from environment variables
    api_key = os.getenv('LLAMA_VISION_FREE_API_KEY')  # Ensure you have this key in your .env file
    if not api_key:
        raise ValueError("LLAMA_VISION_FREE_API_KEY not found in .env file")
    
    # Initialize Together client with the API key
    client = Together(api_key=api_key)

    # Define the model name
    model_name = "meta-llama/Llama-Vision-Free"

    try:
        # Create a chat completion request
        completion = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": "A happy astronaut cat"}]
        )

        # Print the response using the model name
        print(f"{model_name} Response:", completion.choices[0].message.content)
        print("\nConnection test successful! ✅")
        return True
    except Exception as e:
        print(f"Connection test failed! ❌\nError: {str(e)}")
        return False

if __name__ == "__main__":
    test_llama_vision_connection()