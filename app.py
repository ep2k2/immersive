import streamlit as st
import os
import requests  # Import the requests library
from dotenv import load_dotenv
from PIL import Image
from io import BytesIO
import base64
import time  # Import time for sleep functionality
import random  # Import random for generating random seeds

def init_app():
    """Initialize the Streamlit app with basic configuration."""
    st.set_page_config(
        page_title="LLM Image Generator",
        layout="wide"
    )
    st.title("LLM Image Generator")
    return True

def generate_image(prompt):
    """Call the FLUX API to generate an image based on the prompt."""
    load_dotenv()
    api_key = os.getenv('FLUX_SCHNELL_FREE_API_KEY')
    if not api_key:
        raise ValueError("FLUX_SCHNELL_FREE_API_KEY not found in .env file")

    url = "https://api.together.xyz/v1/images/generations"
    
    # Generate a random seed between 0 and 10000
    seed = random.randint(0, 10000)

    payload = {
        "prompt": prompt,
        "model": "black-forest-labs/FLUX.1-schnell-Free",
        "steps": 4,
        "n": 1,
        "height": 400,
        "width": 800,
        "guidance": 3.5,
        "output_format": "png",
        "seed": seed  # Include the random seed in the payload
    }
    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "authorization": f"Bearer {api_key}"
    }

    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        
        print("Response status:", response.status_code)
        print("Response content:", response.text)
        
        response_data = response.json()
        
        if 'data' in response_data and len(response_data['data']) > 0:
            if 'base64' in response_data['data'][0]:
                base64_image = response_data['data'][0]['base64']
                image_data = base64.b64decode(base64_image)
                return Image.open(BytesIO(image_data))
            elif 'url' in response_data['data'][0]:
                image_url = response_data['data'][0]['url']
                img_response = requests.get(image_url)
                img_response.raise_for_status()
                return Image.open(BytesIO(img_response.content))
        
        print("Unexpected response structure:", response_data)
        return None

    except Exception as e:
        print(f"Error during image generation: {str(e)}")
        print("Response content if available:", getattr(response, 'text', 'N/A'))
        return None

def overlay_images(background_image, character_image):
    """Overlay the character image on top of the background image."""
    # Ensure both images are in RGBA mode
    background_image = background_image.convert("RGBA")
    character_image = character_image.convert("RGBA")

    # Get dimensions
    bg_width, bg_height = background_image.size
    char_width, char_height = character_image.size

    # Resize character image if needed (optional)
    character_image = character_image.resize((char_width // 1, char_height // 1
                                              ))

    # Calculate position to align the bottom of the character image with the bottom of the background
    position = ((bg_width - character_image.width) // 2, bg_height - character_image.height)

    # Create a new image for the overlay
    combined_image = Image.new("RGBA", background_image.size)
    combined_image.paste(background_image, (0, 0))  # Paste background
    combined_image.paste(character_image, position, character_image)  # Paste character with transparency

    return combined_image

def main():
    """Main function to run the Streamlit app."""
    init_app()
    
    # User input for character description
    character_description = st.text_area(
        "Enter character description:",
        height=100,
        placeholder="Describe the character you want to generate..."
    )
    
    # User input for background description
    background_description = st.text_area(
        "Enter background description:",
        height=100,
        placeholder="Describe the background you want to generate..."
    )
    
    # Submit button
    if st.button("Generate Images"):
        if character_description and background_description:
            st.info("Processing your prompts...")
            
            # Generate character image
            character_image = generate_image(character_description)  # Call the LLM function for character
            if character_image:
                # Display the character image at a quarter of its original size
                st.image(character_image, caption="Character Image", use_container_width=True)
            else:
                st.error("Failed to generate character image.")
            
            # Wait for 10 seconds before generating the background image
            time.sleep(10)  # Simple wait loop
            
            # Generate background image
            background_image = generate_image(background_description)  # Call the LLM function for background
            if background_image:
                st.image(background_image, caption="Background Image", use_container_width=True)
            else:
                st.error("Failed to generate background image.")

            # Overlay the character on the background
            if character_image and background_image:
                combined_image = overlay_images(background_image, character_image)
                st.image(combined_image, caption="Combined Image", use_container_width=True)
        else:
            st.warning("Please enter both character and background descriptions.")

if __name__ == "__main__":
    main() 