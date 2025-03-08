import streamlit as st
import os
import requests  # Import the requests library
from dotenv import load_dotenv
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
import base64
import time  # Import time for sleep functionality
import random  # Import random for generating random seeds
import cv2
import numpy as np
import textwrap

# Feature flags for development
ENABLE_IMAGE_GENERATION = True  # Set to False to skip image generation (background and character)
ENABLE_AUDIO_GENERATION = True  # Set to False to skip audio generation for dialogue

def init_app():
    """Initialize the Streamlit app with basic configuration."""
    st.set_page_config(
        page_title="LLM Image Generator",
        layout="wide"
    )
    st.title("LLM Image Generator")
    return True

def generate_image(prompt, steps=4, seed=None):
    """Call the FLUX API to generate an image based on the prompt."""
    if not ENABLE_IMAGE_GENERATION:
        print("Image generation is disabled")
        # Return a placeholder image instead
        placeholder = Image.new('RGB', (800, 400), color=(200, 200, 200))
        draw = ImageDraw.Draw(placeholder)
        draw.text((400, 200), "Image Generation Disabled", fill=(0, 0, 0))
        return placeholder
    
    load_dotenv()
    api_key = os.getenv('FLUX_SCHNELL_FREE_API_KEY')
    if not api_key:
        raise ValueError("FLUX_SCHNELL_FREE_API_KEY not found in .env file")

    url = "https://api.together.xyz/v1/images/generations"
    
    # Use provided seed or generate random one if None
    if seed is None:
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
        "seed": seed
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

def create_speech_bubble(image, text, position=(400, 200), max_width=30, offset_y=0, weight='regular'):
    """Create a speech bubble with text on the image."""
    # Create a copy of the image to draw on
    img_with_bubble = image.copy()
    draw = ImageDraw.Draw(img_with_bubble)
    
    # Load the appropriate font based on weight
    font_path = "fonts/NotoSansJP-Regular.ttf"
    if weight == 'bold':
        font_path = "fonts/NotoSansJP-Bold.ttf"
    
    try:
        font = ImageFont.truetype(font_path, 15)
    except (IOError, OSError) as e:
        # Print detailed error for debugging
        print(f"Font error: {e}")
        print(f"Font not found or cannot be used at {font_path}, falling back to default")
        # Try to find any available font that can handle Japanese
        try:
            # Try to use a system font that might support Japanese
            system_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 15)
            font = system_font
        except:
            font = ImageFont.load_default()  # Last resort fallback
    
    # Make bubbles much wider - increase max_width significantly
    max_width_adjusted = 60  # Much wider to avoid unnecessary wrapping
    wrapped_text = textwrap.fill(text, width=max_width_adjusted)
    lines = wrapped_text.split('\n')
    
    # Calculate text size
    line_heights = [draw.textbbox((0, 0), line, font=font)[3] for line in lines]
    text_width = max([draw.textbbox((0, 0), line, font=font)[2] for line in lines])
    text_height = sum(line_heights)
    
    # Calculate bubble size with padding
    padding = 20  # Slightly increased padding for larger font
    bubble_width = text_width + padding * 2
    bubble_height = text_height + padding * 2
    
    # Calculate image dimensions
    img_width, img_height = img_with_bubble.size
    
    # Start position closer to top, with offset based on dialogue index
    # Position centered at 1/3 of the way across the panel
    base_x = img_width // 3
    base_y = 40  # Start closer to the top (was 80)
    
    # Apply only a small horizontal variation to keep bubbles more aligned
    # Use a smaller fraction of the width for horizontal shift
    horizontal_shift = ((offset_y // 60) % 3 - 1) * (img_width // 10)  # Reduced from 1/5 to 1/10
    
    # Adjust position based on offset with smaller vertical increments
    position = (base_x + horizontal_shift, base_y + offset_y)
    
    # Calculate bubble coordinates
    x, y = position
    x = min(max(x, bubble_width // 2 + 20), img_with_bubble.width - bubble_width // 2 - 20)
    y = min(max(y, bubble_height // 2 + 20), img_with_bubble.height - bubble_height - 20)
    
    left = x - bubble_width // 2
    top = y - bubble_height // 2
    right = x + bubble_width // 2
    bottom = y + bubble_height // 2
    
    # Draw bubble with rounded corners
    corner_radius = 20
    draw.rounded_rectangle([left, top, right, bottom], 
                          fill=(255, 255, 255, 230), 
                          outline=(0, 0, 0), 
                          width=2, 
                          radius=corner_radius)
    
    # Draw text in bubble
    y_text = top + padding
    for line in lines:
        line_width = draw.textbbox((0, 0), line, font=font)[2]
        x_text = left + (bubble_width - line_width) // 2
        draw.text((x_text, y_text), line, font=font, fill=(0, 0, 0))
        y_text += line_heights[0]  # Assuming all lines have same height
    
    return img_with_bubble

def init_scene_state():
    """Initialize or reset the scene state in session."""
    default_state = {
        'panels': {
            'top_right': None,
            'bottom_right': None,
            'top_left': None,
            'bottom_left': None
        },
        'panel_count': 0,                 # Track how many panels we've filled
        'current_position': 'top_right',  # Start with top right panel
        'current_stage': 'background',    # Start with background generation
        'scene_background': None,         # Store the scene background for reuse
        'background_image': None,
        'character_image': None,
        'combined_image': None,
        'processing': False,              # Flag to track if we're in the middle of processing
        'dialogue_lines': [],             # Store dialogue lines
        'current_dialogue_index': 0,      # Track which dialogue line we're on
        'dialogue_offsets': {},           # Track vertical offsets for dialogue in each panel
        'dialogue_audio': {}              # Store audio for each dialogue line
    }
    
    if 'scene_state' not in st.session_state:
        st.session_state.scene_state = default_state
    else:
        # Ensure all required keys exist (for backward compatibility)
        for key, value in default_state.items():
            if key not in st.session_state.scene_state:
                st.session_state.scene_state[key] = value

def get_next_panel_position(current_position=None):
    """Get the next panel position in manga reading order."""
    # Japanese manga reading order: top-right → bottom-right → top-left → bottom-left
    order = ['top_right', 'bottom_right', 'top_left', 'bottom_left']
    
    if current_position is None:
        current_position = st.session_state.scene_state['current_position']
    
    current_index = order.index(current_position)
    next_index = (current_index + 1) % len(order)
    return order[next_index]

def process_dialogue(dialogue_line, current_image, current_offset):
    """Process a dialogue line and add it to the current image."""
    # Create a speech bubble with the dialogue
    new_image = create_speech_bubble(current_image, dialogue_line, offset_y=current_offset)
    
    # Generate audio for the dialogue line
    audio_content = generate_speech(dialogue_line)
    
    return new_image, audio_content

def generate_speech(text):
    """Generate speech audio from text using Google's Text-to-Speech API."""
    if not ENABLE_AUDIO_GENERATION:
        print("Audio generation is disabled")
        return None
    
    load_dotenv()
    api_key = os.getenv('GOOGLE_TTS_API_KEY')
    if not api_key:
        print("GOOGLE_TTS_API_KEY not found in .env file")
        return None
    
    url = "https://texttospeech.googleapis.com/v1beta1/text:synthesize"
    
    payload = {
        "audioConfig": {
            "audioEncoding": "LINEAR16",
            "pitch": 1,
            "speakingRate": 1.00
        },
        "input": {
            "text": text
        },
        "voice": {
            "languageCode": "ja-JP",
            "name": "ja-JP-Standard-D"
        }
    }
    
    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": api_key
    }
    
    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        
        print("TTS Response status:", response.status_code)
        
        response_data = response.json()
        
        if 'audioContent' in response_data:
            # Decode the base64 audio content
            audio_content = base64.b64decode(response_data['audioContent'])
            return audio_content
        
        print("Unexpected TTS response structure:", response_data)
        return None
        
    except Exception as e:
        print(f"Error during speech generation: {str(e)}")
        print("Response content if available:", getattr(response, 'text', 'N/A'))
        return None

def play_audio(audio_content):
    """Play audio content in Streamlit."""
    if audio_content:
        # Create a temporary file to store the audio
        audio_file = BytesIO(audio_content)
        st.audio(audio_file, format='audio/wav')

def main():
    """Main function to run the Streamlit app."""
    init_app()
    init_scene_state()
    
    # Create two columns - left for inputs, right for image
    left_col, right_col = st.columns([1, 2])  # 1:2 ratio for column widths
    
    # Check if we need to process dialogue (this happens before UI rendering)
    if 'process_dialogue' in st.session_state and st.session_state.process_dialogue:
        # Get the current dialogue line
        dialogue_index = st.session_state.scene_state['current_dialogue_index']
        if dialogue_index < len(st.session_state.scene_state['dialogue_lines']):
            dialogue_line = st.session_state.scene_state['dialogue_lines'][dialogue_index]
            
            # Get the current panel position
            current_pos = st.session_state.scene_state['current_position']
            
            # Initialize offset tracking for this panel if needed
            if current_pos not in st.session_state.scene_state['dialogue_offsets']:
                st.session_state.scene_state['dialogue_offsets'][current_pos] = 0
            
            # Get current offset for this panel
            current_offset = st.session_state.scene_state['dialogue_offsets'][current_pos]
            
            # Process the dialogue and get audio
            current_image = st.session_state.scene_state['combined_image']
            new_image, audio_content = process_dialogue(dialogue_line, current_image, current_offset)
            
            # Update the combined image with the speech bubble
            st.session_state.scene_state['combined_image'] = new_image
            
            # Store the audio content
            audio_key = f"{current_pos}_{dialogue_index}"
            st.session_state.scene_state['dialogue_audio'][audio_key] = audio_content
            
            # Update the dialogue index and offset
            st.session_state.scene_state['current_dialogue_index'] += 1
            st.session_state.scene_state['dialogue_offsets'][current_pos] += 60
        
        # Reset the processing flags
        st.session_state.process_dialogue = False
        st.session_state.scene_state['processing'] = False
    
    # Continue with the rest of the UI rendering
    with left_col:
        # Move the seed input section to the top - use columns to place label and input side by side
        seed_label_col, seed_input_col = st.columns([1, 3])
        with seed_label_col:
            st.write("Seed:", unsafe_allow_html=True)
        
        with seed_input_col:
            # Check if we need to randomize the seed
            if 'randomize_seed' in st.session_state and st.session_state.randomize_seed:
                # Generate a new random seed
                seed_value = random.randint(0, 10000)
                # Reset the flag
                st.session_state.randomize_seed = False
            else:
                # Use existing seed value
                seed_value = st.session_state.get('seed', 0)
            
            # Use a number input with the seed value (no label)
            st.session_state.seed = st.number_input("Seed", min_value=0, max_value=10000, value=seed_value, format="%d", key="seed_input", label_visibility="collapsed")
        
        # Randomize button directly underneath the seed input
        if st.button("🎲 Randomize"):
            # Set a flag to randomize on next rerun
            st.session_state.randomize_seed = True
            # Force a rerun to update the UI immediately
            st.rerun()

        # Character and background descriptions
        character_description = st.text_area(
            "Enter character description:",
            height=150,
            placeholder="Describe the character you want to generate..."
        )
        
        background_description = st.text_area(
            "Enter background description:",
            height=150,
            placeholder="Describe the background you want to generate..."
        )
        
        # Dialogue input
        dialogue = st.text_area(
            "Enter dialogue:",
            height=100,
            placeholder="Enter character dialogue (one line per bubble)..."
        )

        # Study word focus
        study_word_focus = st.text_area(
            "Study word focus:",
            height=100,
            placeholder="Enter words or phrases to focus on..."
        )
        
        # Generate buttons
        col3, col4, col5 = st.columns(3)
        with col3:
            new_scene_button = st.button("New Scene")
        with col4:
            # Enable next panel button only if we have a completed panel and fewer than 4 panels
            next_panel_button = st.button("Next panel", 
                                         disabled=st.session_state.scene_state['processing'] or 
                                                 st.session_state.scene_state['current_stage'] != 'complete' or
                                                 st.session_state.scene_state['panel_count'] >= 3)
        with col5:
            # Process dialogue input and store in session state
            if dialogue:
                # Split dialogue by newlines and filter out empty lines
                dialogue_lines = [line.strip() for line in dialogue.split('\n') if line.strip()]
                
                # Only reset dialogue index if the dialogue content has changed
                current_dialogue = st.session_state.scene_state.get('dialogue_lines', [])
                if dialogue_lines != current_dialogue:
                    st.session_state.scene_state['dialogue_lines'] = dialogue_lines
                    st.session_state.scene_state['current_dialogue_index'] = 0
                    # Also reset dialogue offsets when dialogue changes
                    st.session_state.scene_state['dialogue_offsets'] = {}
            
            # Enable next dialogue button only if we have dialogue lines and a completed panel
            has_unused_dialogue = (st.session_state.scene_state['current_dialogue_index'] < 
                                  len(st.session_state.scene_state['dialogue_lines']))
            
            next_dialogue_button = st.button("Next dialogue", 
                                           disabled=st.session_state.scene_state['processing'] or 
                                                   st.session_state.scene_state['current_stage'] != 'complete' or
                                                   not has_unused_dialogue)

    # Show processing message in left column when generating
    if st.session_state.scene_state['processing']:
        st.info("Processing your prompts...")
    
    with right_col:
        # Create layout for manga panels
        scene_container = st.container()
        with scene_container:
            # Create a 2x2 grid for panels in the correct order
            row1 = st.columns(2)
            row2 = st.columns(2)
            
            # Get columns in the right order for manga reading
            tr_col = row1[0]  # Top-right (first position)
            br_col = row2[0]  # Bottom-right (second position)
            tl_col = row1[1]  # Top-left (third position)
            bl_col = row2[1]  # Bottom-left (fourth position)
            
            # Display panels in manga order
            with tr_col:
                if st.session_state.scene_state['panels']['top_right'] is not None:
                    st.image(st.session_state.scene_state['panels']['top_right'], use_container_width=True)
            with br_col:
                if st.session_state.scene_state['panels']['bottom_right'] is not None:
                    st.image(st.session_state.scene_state['panels']['bottom_right'], use_container_width=True)
            with tl_col:
                if st.session_state.scene_state['panels']['top_left'] is not None:
                    st.image(st.session_state.scene_state['panels']['top_left'], use_container_width=True)
            with bl_col:
                if st.session_state.scene_state['panels']['bottom_left'] is not None:
                    st.image(st.session_state.scene_state['panels']['bottom_left'], use_container_width=True)
            
            # Display current working image in the appropriate panel
            current_position = st.session_state.scene_state['current_position']
            current_image = None
            
            if st.session_state.scene_state['current_stage'] == 'background':
                current_image = st.session_state.scene_state['background_image']
            elif st.session_state.scene_state['current_stage'] in ['character', 'complete']:
                current_image = st.session_state.scene_state['combined_image']
            
            if current_image is not None:
                # Display in the current panel position
                if current_position == 'top_right':
                    with tr_col:
                        st.image(current_image, use_container_width=True)
                elif current_position == 'bottom_right':
                    with br_col:
                        st.image(current_image, use_container_width=True)
                elif current_position == 'top_left':
                    with tl_col:
                        st.image(current_image, use_container_width=True)
                elif current_position == 'bottom_left':
                    with bl_col:
                        st.image(current_image, use_container_width=True)
            
            # Play the most recent audio if available
            if st.session_state.scene_state['current_dialogue_index'] > 0:
                dialogue_index = st.session_state.scene_state['current_dialogue_index'] - 1
                audio_key = f"{current_position}_{dialogue_index}"
                
                if audio_key in st.session_state.scene_state['dialogue_audio']:
                    audio_content = st.session_state.scene_state['dialogue_audio'][audio_key]
                    if audio_content:
                        # Create a container for the audio player
                        audio_container = st.container()
                        with audio_container:
                            st.write("Audio for current dialogue:")
                            play_audio(audio_content)

    # Handle generation
    if new_scene_button:
        if character_description and background_description:
            # Process dialogue input for the new scene
            if dialogue:
                # Split dialogue by newlines and filter out empty lines
                dialogue_lines = [line.strip() for line in dialogue.split('\n') if line.strip()]
            else:
                dialogue_lines = []
                
            # Reset scene state for new scene
            st.session_state.scene_state = {
                'panels': {
                    'top_right': None,
                    'bottom_right': None,
                    'top_left': None,
                    'bottom_left': None
                },
                'panel_count': 0,
                'current_position': 'top_right',
                'current_stage': 'background',
                'scene_background': None,
                'background_image': None,
                'character_image': None,
                'combined_image': None,
                'processing': True,
                'dialogue_lines': dialogue_lines,
                'current_dialogue_index': 0,
                'dialogue_offsets': {},
                'dialogue_audio': {}
            }
            
            # Start generating the first panel
            st.rerun()
            
    elif next_panel_button:
        if character_description and background_description:
            # Save the current completed panel
            current_pos = st.session_state.scene_state['current_position']
            st.session_state.scene_state['panels'][current_pos] = st.session_state.scene_state['combined_image']
            st.session_state.scene_state['panel_count'] += 1
            
            # Move to the next panel position
            next_pos = get_next_panel_position()
            st.session_state.scene_state['current_position'] = next_pos
            st.session_state.scene_state['current_stage'] = 'character'  # Skip background generation
            st.session_state.scene_state['background_image'] = st.session_state.scene_state['scene_background']  # Reuse background
            st.session_state.scene_state['character_image'] = None
            st.session_state.scene_state['combined_image'] = None
            st.session_state.scene_state['processing'] = True
            
            st.rerun()
        else:
            st.warning("Please enter both character and background descriptions.")
    
    elif next_dialogue_button:
        if st.session_state.scene_state['combined_image'] is not None:
            # Set flags to process dialogue on next rerun
            st.session_state.process_dialogue = True
            st.session_state.scene_state['processing'] = True
            
            # Force a rerun to process the dialogue
            st.rerun()
    
    # Process the current stage if we're in processing mode
    if st.session_state.scene_state['processing']:
        process_current_stage(character_description, background_description, st.session_state.seed)

def process_current_stage(character_description, background_description, seed):
    """Process the current stage of scene generation."""
    current_stage = st.session_state.scene_state['current_stage']
    
    if current_stage == 'background':
        # Generate background image only for the first panel in a scene
        background_image = generate_image(background_description, steps=4, seed=seed)
        if background_image:
            # Apply Gaussian blur for combined view
            blurred_background = apply_gaussian_blur(background_image)
            
            # Store the background image for this scene
            st.session_state.scene_state['scene_background'] = blurred_background
            st.session_state.scene_state['background_image'] = blurred_background
            
            # Move directly to character stage
            st.session_state.scene_state['current_stage'] = 'character'
            
            # Wait for a moment before generating character (reduced if image generation is disabled)
            if ENABLE_IMAGE_GENERATION:
                time.sleep(10)
            else:
                time.sleep(1)  # Shorter wait time when image generation is disabled
            
            st.rerun()
        else:
            st.error("Failed to generate background image.")
            st.session_state.scene_state['processing'] = False
    
    elif current_stage == 'character':
        # Generate character image
        character_image = generate_image(character_description, steps=4, seed=seed)
        if character_image:
            # Get the background image
            background_image = st.session_state.scene_state['background_image']
            
            # Create combined image
            combined_image = overlay_images(background_image, character_image)
            st.session_state.scene_state['combined_image'] = combined_image
            st.session_state.scene_state['character_image'] = character_image
            
            # Mark as complete
            st.session_state.scene_state['current_stage'] = 'complete'
            st.session_state.scene_state['processing'] = False
            
            st.rerun()
        else:
            st.error("Failed to generate character image.")
            st.session_state.scene_state['processing'] = False

if __name__ == "__main__":
    main() 