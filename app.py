import streamlit as st
import os
import requests  # Import the requests library
from dotenv import load_dotenv
from PIL import Image
from io import BytesIO
import base64
import time  # Import time for sleep functionality
import random  # Import random for generating random seeds
import cv2
import numpy as np

def init_app():
    """Initialize the Streamlit app with basic configuration."""
    st.set_page_config(
        page_title="LLM Image Generator",
        layout="wide"
    )
    st.title("LLM Image Generator")
    return True

def generate_image(prompt, steps=4):
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
        "steps": steps,
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

def make_background_transparent_flood_fill(image):
    """Convert background to transparent using OpenCV's floodFill."""
    # Convert the image to a NumPy array
    image_np = np.array(image)

    # Convert to RGB if the image has an alpha channel (RGBA)
    if image_np.shape[2] == 4:  # Check if the image has 4 channels (RGBA)
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2RGB)  # Convert to RGB

    # Create a mask for flood fill
    h, w = image_np.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)

    # Perform flood fill from the top-left corner (0, 0)
    cv2.floodFill(image_np, mask, (0, 0), (0, 0, 0), loDiff=(10, 10, 10), upDiff=(10, 10, 10))

    # Perform flood fill from the bottom-right corner (w-1, h-1)
    cv2.floodFill(image_np, mask, (w-1, h-1), (0, 0, 0), loDiff=(10, 10, 10), upDiff=(10, 10, 10))

    # Convert back to RGBA
    image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2RGBA)  # Convert back to RGBA

    # Set the filled areas to transparent
    filled_mask = mask[1:-1, 1:-1] > 0  # Get the mask of filled areas
    image_np[filled_mask] = (0, 0, 0, 0)  # Set filled pixels to transparent

    return Image.fromarray(image_np, 'RGBA')

def make_background_transparent_masking(image):
    """Convert pure white background to transparent using masking."""
    # Convert the image to a NumPy array
    image_np = np.array(image)

    # Convert to RGBA if the image has an alpha channel (RGBA)
    if image_np.shape[2] == 4:  # Check if the image has 4 channels (RGBA)
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2RGB)  # Convert to RGB

    # Convert back to RGBA
    image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2RGBA)  # Convert back to RGBA

    # Create a mask for pure white pixels
    white_mask = (image_np[:, :, 0] == 255) & (image_np[:, :, 1] == 255) & (image_np[:, :, 2] == 255)

    # Set pure white pixels to transparent
    image_np[white_mask] = (0, 0, 0, 0)  # Set filled pixels to transparent

    return Image.fromarray(image_np, 'RGBA')

def alpha_blend(foreground, background):
    """Blend foreground image with background using alpha blending."""
    # Ensure both images are in RGBA format
    foreground = foreground.convert("RGBA")
    background = background.convert("RGBA")

    # Convert images to NumPy arrays
    fg_np = np.array(foreground)
    bg_np = np.array(background)

    # Extract the alpha channel from the foreground
    alpha_channel = fg_np[:, :, 3] / 255.0  # Normalize alpha to [0, 1]

    # Create an output image
    output = np.zeros_like(bg_np)

    # Perform alpha blending
    for c in range(0, 3):  # Loop over RGB channels
        output[:, :, c] = (alpha_channel * fg_np[:, :, c] + (1 - alpha_channel) * bg_np[:, :, c])

    # Set the alpha channel of the output image
    output[:, :, 3] = 255  # Set alpha to fully opaque

    return Image.fromarray(output, 'RGBA')

def overlay_images(background_image, character_image):
    """Overlay the character image on top of the background image using alpha blending."""
    # Ensure both images are in RGBA mode
    background_image = background_image.convert("RGBA")
    character_image = make_background_transparent_flood_fill(character_image)  # Use flood fill for transparency

    # Get dimensions of the background image
    bg_width, bg_height = background_image.size

    # Create a new canvas with the desired dimensions (e.g., 800x400)
    canvas_width = 800  # Desired canvas width
    canvas_height = 400  # Desired canvas height
    canvas = Image.new("RGBA", (canvas_width, canvas_height), (255, 255, 255, 0))  # Transparent background

    # Resize background image to fit canvas
    background_image_resized = background_image.resize((canvas_width, canvas_height))  # Resize background to fit canvas

    # Paste the background image onto the canvas
    canvas.paste(background_image_resized, (0, 0))  # Paste background at the top-left corner

    # Calculate position to align the bottom of the character image with the bottom of the canvas
    # Position character closer to the right edge (4/5 of the way across)
    position_x = int(canvas_width * (2/3)) - (character_image.width // 2)  # Position two-thirds of the way to the right
    position_y = canvas_height - character_image.height  # Align bottom

    # Create a new canvas for the character at the correct position
    character_canvas = Image.new("RGBA", (canvas_width, canvas_height), (255, 255, 255, 0))
    character_canvas.paste(character_image, (position_x, position_y))

    # Overlay the positioned character on the canvas using alpha blending
    combined_image = alpha_blend(character_canvas, canvas)

    return combined_image

def apply_gaussian_blur(image):
    """Apply Gaussian blur to an image."""
    # Ensure image is in RGBA mode
    image = image.convert("RGBA")
    
    # Convert image to NumPy array for Gaussian blur
    img_np = np.array(image)
    
    # Apply Gaussian blur
    blurred_np = cv2.GaussianBlur(img_np, (5, 5), 0)  # Adjust kernel size as needed
    
    # Convert back to PIL Image
    blurred_image = Image.fromarray(blurred_np, 'RGBA')
    
    return blurred_image

def fade_transition(placeholder, old_image, new_image, steps=5):
    """Create a fade transition between two images."""
    for alpha in range(0, steps + 1):
        # Calculate the alpha value for blending
        alpha_value = alpha / steps
        
        # Create a blended image
        blended_np = np.array(old_image) * (1 - alpha_value) + np.array(new_image) * alpha_value
        blended_image = Image.fromarray(blended_np.astype(np.uint8), 'RGBA')
        
        # Update the placeholder with the blended image
        placeholder.image(blended_image, use_container_width=True)
        
        # Shorter delay for faster animation (0.5 seconds total for 5 steps = 0.1 seconds per step)
        time.sleep(0.1)  # Updated delay for faster transition

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
    
    # Create a placeholder for the image display
    image_placeholder = st.empty()
    
    # Submit button
    if st.button("Generate Images"):
        if character_description and background_description:
            st.info("Processing your prompts...")
            
            # Generate background image
            background_image = generate_image(background_description, steps=4)  # Call with steps=4 for background
            if background_image:
                # Apply Gaussian blur to the background image
                blurred_background = apply_gaussian_blur(background_image)
                
                # Display the blurred background image in the placeholder
                image_placeholder.image(blurred_background, caption="Background Image", use_container_width=True)
                
                # Store the blurred background for fade transition
                current_image = blurred_background
            else:
                st.error("Failed to generate background image.")
                return
            
            # Wait for 10 seconds before generating the character image
            time.sleep(10)  # Simple wait loop
            
            # Generate character image
            character_image = generate_image(character_description, steps=4)  # Call with steps=4 for character
            if character_image:
                # Overlay the character on the background
                combined_image = overlay_images(blurred_background, character_image)
                
                # Apply fade transition from background to combined image
                try:
                    fade_transition(image_placeholder, current_image, combined_image)
                except Exception as e:
                    # Fallback if fade transition fails
                    st.warning(f"Fade transition failed: {str(e)}")
                    image_placeholder.image(combined_image, caption="Combined Image", use_container_width=True)
            else:
                st.error("Failed to generate character image.")
        else:
            st.warning("Please enter both character and background descriptions.")

if __name__ == "__main__":
    main() 