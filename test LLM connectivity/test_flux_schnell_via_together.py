import os
import requests  # Import the requests library
from dotenv import load_dotenv
from PIL import Image
from io import BytesIO
import base64

def test_flux_schnell_connection():
    # Load environment variables
    load_dotenv()
    
    # Access the API key from environment variables
    api_key = os.getenv('FLUX_SCHNELL_FREE_API_KEY')  # Ensure you have this key in your .env file
    if not api_key:
        raise ValueError("FLUX_SCHNELL_FREE_API_KEY not found in .env file")
    
    # Define the API endpoint and payload
    url = "https://api.together.xyz/v1/images/generations"
    payload = {
       "prompt": "A happy astronaut cat",
       "model": "black-forest-labs/FLUX.1-schnell-Free",
       "n": 1,
       "height": 1024,
       "width": 1024,
       "guidance": 3.5,
       "response_format": "base64",  # Request base64 encoded image
       "output_format": "png"
    }
    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "authorization": f"Bearer {api_key}"  # Use the API key for authorization
    }

    try:
        # Make the POST request to the image generation endpoint
        response = requests.post(url, json=payload, headers=headers)

        # Check if the request was successful
        response.raise_for_status()  # Raise an error for bad responses

        # Print the response
        print("FLUX Response:", response.json())  # Print the JSON response
        print("\nConnection test successful! ✅")

        # Assuming you have the base64 image data from the response
        base64_image = response.json()['data'][0]['base64']  # Get the base64 image data from the response

        # Decode the base64 string
        image_data = base64.b64decode(base64_image)

        # Create an image from the decoded data
        image = Image.open(BytesIO(image_data))

        # Save the image to a file
        image.save("happy_astronaut_cat.png")
        print("Image saved as happy_astronaut_cat.png")

        return True 
    
    except Exception as e:  # Added exception handling
        print(f"Connection test failed! ❌\nError: {str(e)}")
        print("Response Text:", response.text)  # Print the response text for debugging
        return False  # Ensure return is within the function

if __name__ == "__main__":
    test_flux_schnell_connection() 