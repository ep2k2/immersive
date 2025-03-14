import streamlit as st
import os
import requests  # Import the requests library
import re  # Import regular expressions module
from dotenv import load_dotenv
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
import time
import random
import base64
import cv2
import numpy as np
import textwrap
import google.generativeai as genai
import json

# Feature flags for development
ENABLE_IMAGE_GENERATION = True # Set to False to skip image generation (background and character)
ENABLE_AUDIO_GENERATION = True # Set to False to skip audio generation for dialogue

DEBUG_MODE = True  # Set to True to enable debug output

def add_debug_message(message):
    """Print debug message to terminal if debug mode is enabled."""
    if DEBUG_MODE:
        print("DEBUG:", message)  # Print to terminal for reference

def init_app():
    """Initialize the Streamlit app with basic configuration."""
    st.set_page_config(
        page_title="日常のスケッチ-EXP",
        layout="wide"
    )
    st.title("日常のスケッチ-EXP 📅🖼️")
    return True

def generate_image(prompt, steps=4, seed=None):
    """Call the FLUX API to generate an image based on the prompt."""
    global last_image_generation_time
    
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

    # Implement rate limiting - ensure at least 10 seconds between API calls
    current_time = time.time()
    time_since_last_call = current_time - last_image_generation_time
    
    if time_since_last_call < 10:
        wait_time = 10 - time_since_last_call
        print(f"Rate limiting: Waiting {wait_time:.2f} seconds before next image generation")
        time.sleep(wait_time)
    
    # Update the last image generation time
    last_image_generation_time = time.time()
    
    # Retry logic for 429 errors
    max_retries = 3
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            response = requests.post(url, json=payload, headers=headers)
            
            # Check for 429 error (Too Many Requests)
            if response.status_code == 429:
                retry_count += 1
                if retry_count < max_retries:
                    wait_time = 5  # Wait 5 seconds before retrying
                    print(f"Received 429 error. Waiting {wait_time} seconds before retry {retry_count}/{max_retries}")
                    time.sleep(wait_time)
                    continue
            
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
            
        except requests.exceptions.HTTPError as e:
            if response.status_code == 429 and retry_count < max_retries:
                # This case is handled above
                pass
            else:
                print(f"HTTP Error during image generation: {str(e)}")
                print("Response content if available:", getattr(response, 'text', 'N/A'))
                return None
                
        except Exception as e:
            print(f"Error during image generation: {str(e)}")
            print("Response content if available:", getattr(response, 'text', 'N/A'))
            return None
    
    # If we've exhausted all retries
    print("Maximum retries reached. Unable to generate image.")
    return None

last_image_generation_time = 0

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
    # Ensure both images are in RGBA mode and not None
    if background_image is None or character_image is None:
        add_debug_message("Error: Received None image in overlay_images function")
        # Return a placeholder image instead
        placeholder = Image.new('RGBA', (800, 400), color=(200, 200, 200, 255))
        draw = ImageDraw.Draw(placeholder)
        draw.text((400, 200), "Image Processing Error", fill=(0, 0, 0))
        return placeholder
        
    # Continue with normal processing
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

def create_speech_bubble(image, text, position=(400, 200), max_width=30, offset_y=0, weight='regular', is_ai=True):
    """Create a speech bubble with text on the image."""
    # Create a copy of the image to draw on
    img_with_bubble = image.copy()
    draw = ImageDraw.Draw(img_with_bubble)
    
    # Load the appropriate font based on weight
    font_path = "fonts/NotoSansJP-Regular.ttf"
    if weight == 'bold':
        font_path = "fonts/NotoSansJP-Bold.ttf"
    
    try:
        # Further reduced font size (was 13)
        font = ImageFont.truetype(font_path, 11)
    except (IOError, OSError) as e:
        # Print detailed error for debugging
        print(f"Font error: {e}")
        print(f"Font not found or cannot be used at {font_path}, falling back to default")
        # Try to find any available font that can handle Japanese
        try:
            # Try to use a system font that might support Japanese
            system_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11)
            font = system_font
        except:
            font = ImageFont.load_default()  # Last resort fallback
    
    max_width_adjusted = 60  # Much wider to avoid unnecessary wrapping
    wrapped_text = textwrap.fill(text, width=max_width_adjusted)
    lines = wrapped_text.split('\n')
    
    # Calculate text size
    line_heights = [draw.textbbox((0, 0), line, font=font)[3] for line in lines]
    text_width = max([draw.textbbox((0, 0), line, font=font)[2] for line in lines])
    text_height = sum(line_heights)
    
    padding = 8
    bubble_width = text_width + padding * 2
    bubble_height = text_height + padding * 2
    
    # Calculate image dimensions
    img_width, img_height = img_with_bubble.size
    
    # Set bubble position based on whether it's AI or user dialogue
    if is_ai:
        # AI dialogue: left aligned very near the left border
        base_x = bubble_width // 2 + 20  # Very close to left edge
    else:
        # User dialogue: right edge aligned with center line of panel
        base_x = (img_width // 2) - (bubble_width // 2)  # Right edge at center line
    
    base_y = 40  # Start closer to the top
    
    # Apply only a small horizontal variation to keep bubbles more aligned
    if is_ai:
        # Less horizontal variation for AI bubbles to keep them aligned left
        horizontal_shift = ((offset_y // 60) % 2) * (img_width // 30)
    else:
        # Less horizontal variation for user bubbles too
        horizontal_shift = ((offset_y // 60) % 2) * (img_width // 30)
    
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
    
    # Set bubble and text colors based on whether it's AI or user dialogue
    if is_ai:
        # AI dialogue: white fill with jade text/outline
        bubble_color = (255, 255, 255, 230)  # White with transparency
        text_color = (0, 100, 80)  # Jade text
        outline_color = (0, 80, 60)  # Darker jade for outline
    else:
        # User dialogue: white fill with dark orange text/outline
        bubble_color = (255, 255, 255, 230)  # White with transparency
        text_color = (180, 80, 0)  # Dark orange text
        outline_color = (150, 60, 0)  # Darker orange for outline
        
    # Draw bubble with rounded corners
    corner_radius = 12  # Further reduced corner radius (was 15)
    draw.rounded_rectangle([left, top, right, bottom], 
                           fill=bubble_color, 
                           outline=outline_color, 
                           width=2, 
                           radius=corner_radius)
    
    # Draw text in bubble
    y_text = top + padding
    for line in lines:
        line_width = draw.textbbox((0, 0), line, font=font)[2]
        
        # Text alignment: center for both AI and user bubbles
        x_text = left + (bubble_width - line_width) // 2  # Center-aligned within the bubble
            
        draw.text((x_text, y_text), line, font=font, fill=text_color)
        y_text += line_heights[0]  # Assuming all lines have same height
    
    return img_with_bubble

def init_scene_state():
    """Initialize or reset the scene state in session."""
    default_state = {
        'panels': {},  # Empty dictionary to store panels
        'panel_count': 0,
        'current_position': 'panel_0',  # Simplified position naming
        'current_panel_number': 0,  # Track the current panel number (0 for setup, 1+ for content)
        'panel_data': {},  # Store panel-specific data (character images, dialogue, etc.)
        'current_stage': 'background',
        'scene_background': None,
        'background_image': None,
        'character_image': None,
        'combined_image': None,
        'processing': False,
        'dialogue_lines': [],
        'current_dialogue_index': 0,
        'dialogue_offsets': {},
        'dialogue_audio': {},
        'character_gender': None  # Track character gender for voice selection
    }
    
    try:
        if 'scene_state' not in st.session_state:
            st.session_state.scene_state = default_state
        else:
            # Ensure all required keys exist (for backward compatibility)
            for key, value in default_state.items():
                if key not in st.session_state.scene_state:
                    st.session_state.scene_state[key] = value
    except RuntimeError as e:
        # This catches the ScriptRunContext warning
        add_debug_message(f"Note: {str(e)}")
        # The session state will be properly initialized when the app fully loads

def process_dialogue(dialogue_line, current_image, current_offset, is_ai=True):
    """Process a dialogue line and add it to the current image.
    
    Args:
        dialogue_line: The text to display in the speech bubble
        current_image: The current panel image
        current_offset: Vertical offset for positioning the bubble
        is_ai: Whether the dialogue is from the AI (True) or user (False)
    
    Returns:
        Tuple of (new_image, audio_content)
    """
    # Create a speech bubble with the dialogue, specifying if it's AI or user
    new_image = create_speech_bubble(current_image, dialogue_line, offset_y=current_offset, is_ai=is_ai)
    
    # Generate audio for the dialogue line (only for AI dialogue)
    audio_content = None
    if is_ai:
        audio_content = generate_speech(dialogue_line)
    
    return new_image, audio_content

def add_dialogue_to_panel(dialogue_line, is_ai=True):
    """Add a dialogue line to the current panel.
    
    Args:
        dialogue_line: The text to display in the speech bubble
        is_ai: Whether the dialogue is from the AI (True) or user (False)
    """
    # Only proceed if we have a valid panel and dialogue line
    if not dialogue_line or not st.session_state.scene_state.get('combined_image'):
        return
    
    # Get the current panel position
    current_pos = st.session_state.scene_state.get('current_position')
    if not current_pos:
        return
    
    # Initialize offset tracking for this panel if needed
    if current_pos not in st.session_state.scene_state['dialogue_offsets']:
        st.session_state.scene_state['dialogue_offsets'][current_pos] = 0
    
    # Get current offset for this panel
    current_offset = st.session_state.scene_state['dialogue_offsets'][current_pos]
    
    # For AI responses, split the dialogue by Japanese periods and process each part
    if is_ai and "。" in dialogue_line:
        # Generate audio for the full dialogue line first (we only want one audio file)
        audio_content = generate_speech(dialogue_line) if ENABLE_AUDIO_GENERATION else None
        
        # Split the dialogue into parts
        dialogue_parts = split_dialogue_by_japanese_period(dialogue_line)
        add_debug_message(f"Split dialogue into {len(dialogue_parts)} parts: {dialogue_parts}")
        
        # Process each part separately for display
        current_image = st.session_state.scene_state['combined_image']
        for part in dialogue_parts:
            if part.strip():
                # Create speech bubble for this part
                current_image = create_speech_bubble(current_image, part, offset_y=current_offset, is_ai=is_ai)
                
                # Increment the offset for the next bubble
                current_offset += 45
        
        # Update the combined image with all speech bubbles
        st.session_state.scene_state['combined_image'] = current_image
        
        # Store the audio content if available
        if audio_content:
            dialogue_index = st.session_state.scene_state.get('current_dialogue_index', 0)
            audio_key = f"{current_pos}_{dialogue_index}"
            st.session_state.scene_state['dialogue_audio'][audio_key] = audio_content
            # Update the dialogue index
            st.session_state.scene_state['current_dialogue_index'] = dialogue_index + 1
        
        # Update the final offset for the panel
        st.session_state.scene_state['dialogue_offsets'][current_pos] = current_offset
    else:
        # Process the dialogue as a single unit (original behavior)
        current_image = st.session_state.scene_state['combined_image']
        new_image, audio_content = process_dialogue(dialogue_line, current_image, current_offset, is_ai=is_ai)
        
        # Update the combined image with the speech bubble
        st.session_state.scene_state['combined_image'] = new_image
        
        # Store the audio content if available (only for AI dialogue)
        if audio_content and is_ai:
            dialogue_index = st.session_state.scene_state.get('current_dialogue_index', 0)
            audio_key = f"{current_pos}_{dialogue_index}"
            st.session_state.scene_state['dialogue_audio'][audio_key] = audio_content
            # Update the dialogue index
            st.session_state.scene_state['current_dialogue_index'] = dialogue_index + 1
        
        # Increment the offset for the next dialogue bubble with a smaller increment for slight overlap
        st.session_state.scene_state['dialogue_offsets'][current_pos] += 45

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
    
    # Get character gender and apply voice variation
    character_gender = st.session_state.scene_state.get('character_gender')
    
    # Debug log the character gender
    if character_gender:
        print(f"DEBUG: Using character gender: {character_gender}")
    else:
        print("DEBUG: WARNING - Character gender is missing! This should be provided by the LLM.")
        # Fall back to female if missing, but this should not happen
        character_gender = 'female'
    
    # Select randomly between the two voices for each gender
    if character_gender == "female":
        voice_name = random.choice(["ja-JP-Standard-A", "ja-JP-Standard-C"])
    else:
        voice_name = random.choice(["ja-JP-Standard-B", "ja-JP-Standard-D"])
    
    # Add random pitch variation within ±15%
    pitch_variation = random.uniform(-0.15, 0.15)
    
    payload = {
        "audioConfig": {
            "audioEncoding": "LINEAR16",
            "pitch": pitch_variation,
            "speakingRate": 1.00
        },
        "input": {
            "text": text
        },
        "voice": {
            "languageCode": "ja-JP",
            "name": voice_name
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
        st.audio(audio_file, format='audio/wav', autoplay=True)  # Set autoplay to True

def parse_llm_response(raw_response):
    """Extract and parse the JSON from the LLM response."""
    try:
        # First, try to find JSON between triple backticks
        json_pattern = r'```(?:json)?\s*({[\s\S]*?})\s*```'
        json_match = re.search(json_pattern, raw_response)
        
        if json_match:
            json_string = json_match.group(1)
            add_debug_message(f"Extracted JSON from code block: {json_string}")
            return json.loads(json_string)
        
        # If that fails, try to find the outermost JSON object
        start_index = raw_response.find('{')
        if start_index == -1:
            add_debug_message("No JSON object found in response")
            return None
            
        # Find matching closing brace
        brace_count = 0
        end_index = -1
        
        for i in range(start_index, len(raw_response)):
            if raw_response[i] == '{':
                brace_count += 1
            elif raw_response[i] == '}':
                brace_count -= 1
                if brace_count == 0:
                    end_index = i + 1
                    break
        
        if end_index == -1:
            add_debug_message("No matching closing brace found")
            return None
            
        # Extract the JSON string
        json_string = raw_response[start_index:end_index]
        
        add_debug_message(f"Extracted JSON String: {json_string}")
        
        # Parse the JSON string into a Python dictionary
        return json.loads(json_string)
    
    except Exception as e:
        add_debug_message(f"Error parsing JSON: {e}")
        return None  # Return None if parsing fails

def play_audio_with_autoplay(audio_content):
    """Play audio content in Streamlit with autoplay."""
    if audio_content:
        # Create an HTML audio element with autoplay
        audio_bytes = base64.b64encode(audio_content).decode()
        audio_html = f"""
        <audio autoplay controls>
            <source src="data:audio/wav;base64,{audio_bytes}" type="audio/wav">
            Your browser does not support the audio element.
        </audio>
        <script>
            // This ensures the audio only autoplays once per session
            const audioElements = document.getElementsByTagName("audio");
            const latestAudio = audioElements[audioElements.length - 1];
            
            // Check if this audio has already been played
            const audioId = "{hash(audio_bytes) % 10000000}";
            if (!sessionStorage.getItem("played_" + audioId)) {{
                latestAudio.play().catch(e => console.log("Autoplay prevented:", e));
                sessionStorage.setItem("played_" + audioId, "true");
            }}
        </script>
        """
        st.markdown(audio_html, unsafe_allow_html=True)

def split_dialogue_by_japanese_period(text):
    """Split dialogue text using Japanese full-stop period character "。".
    
    Args:
        text: The dialogue text to split
        
    Returns:
        List of dialogue parts split by Japanese period
    """
    # Split the text by Japanese period "。"
    parts = text.split("。")
    
    # Filter out empty parts and add the period back to non-empty parts
    result = []
    for part in parts:
        if part.strip():
            result.append(part + "。")
    
    # If the original text doesn't end with a period, remove it from the last part
    if not text.endswith("。") and result:
        result[-1] = result[-1][:-1]
    
    return result

def main():
    """Main function to run the Streamlit app."""
    init_app()
    init_scene_state()
    
    # Add a debug message for the current time
    add_debug_message(f"Debug: Current time: {time.strftime('%H:%M:%S')}")
    
    # Check for generate_panel_1 flag first (this takes priority)
    if 'generate_panel_1' in st.session_state and st.session_state.generate_panel_1:
        # Reset the flag immediately to prevent double execution
        st.session_state.generate_panel_1 = False
        add_debug_message("Automatically generating Panel 1...")
        
        # Get necessary data from session state
        scenario_description_english = st.session_state.get('scenario_description_english', '')
        introduction_english = st.session_state.get('introduction_english', '')
        background_image_prompt_english = st.session_state.get('background_image_prompt_english', '')
        word_list = st.session_state.get('word_list', [])
        
        # Create a prompt for Panel 1
        panel1_prompt = f"""NEW SESSION: Ignore all previous context. You are acting as a conversational partner in a Japanese language learning app. Your task is to generate Panel 1 for the conversation scenario described below.

**Output only JSON in your response, exactly as specified in the format below. Do not include any additional content outside the JSON format.**

Here's the scenario information from Panel 0:
- Scenario: {scenario_description_english}
- Introduction: {introduction_english}
- Background: {background_image_prompt_english}
- Word list: {', '.join(word_list)}

Generate Panel 1 with a character image prompt and the first dialogue line in Japanese.
Use at least one study word naturally in this first dialogue line.

Your character image prompt must follow these requirements:
- Focus ONLY on the character - do NOT include any environmental details or positioning
- Specify the character's age, gender, and appearance (height, build, glasses, etc.)
- State that the character is Japanese unless the scenario indicates otherwise
- Describe clothing in general terms without colors or patterns
- Briefly describe the character's current action or emotion
- NEVER include colors (e.g., "brown", "red", "blue"), patterns, or style information

Format your response as this JSON object:
```json
{{
    "panel-number": 1,
    "exchanges": [
        {{
            "character-image-prompt-english": "[Detailed character description including appearance and emotion]",
            "dialogue-line-japanese": "[First dialogue line in Japanese using at least one study word]"
        }}
    ]
}}
```"""
        
        # Call the API to generate Panel 1
        panel1_response = call_gemini_api(panel1_prompt)
        add_debug_message(f"Raw Panel 1 response: {panel1_response}")
        
        # Parse the Panel 1 response
        panel1_data = parse_llm_response(panel1_response)
        add_debug_message(f"Parsed Panel 1 data: {panel1_data}")
        
        if panel1_data is not None and panel1_data.get("panel-number") == 1:
            # Store Panel 1 data in session state
            add_debug_message(f"Successfully generated Panel 1: {panel1_data}")
            
            # Extract the character image prompt and dialogue line
            if "exchanges" in panel1_data and len(panel1_data["exchanges"]) > 0:
                first_exchange = panel1_data["exchanges"][0]
                character_image_prompt = first_exchange.get("character-image-prompt-english", "")
                
                # Store in session state
                st.session_state.character_image_prompt_english = character_image_prompt
                
                # Extract dialogue line from the first exchange
                dialogue_line = first_exchange.get("dialogue-line-japanese", "")
                
                # Update the scene state to panel 1
                st.session_state.scene_state["current_position"] = "panel_1"
                st.session_state.scene_state["current_panel_number"] = 1
                
                # Process panel 1 using the new panel processing function
                st.session_state.scene_state["processing"] = True
                process_panel(1, panel1_data, character_description=character_image_prompt, seed=st.session_state.seed)
        else:
            add_debug_message("Failed to generate Panel 1 or response was invalid")
        
        # Force a rerun to update the UI
        st.rerun()
    
    # Check for process_llm_response flag (only if generate_panel_1 wasn't processed)
    elif 'process_llm_response' in st.session_state and st.session_state.process_llm_response:
        # Reset the flag immediately to prevent infinite loops
        st.session_state.process_llm_response = False
        
        # Check if we have an LLM response to process
        if 'llm_response' in st.session_state and st.session_state.llm_response:
            raw_response = st.session_state.llm_response
            
            # Debug the raw LLM response
            add_debug_message("Checking LLM Response")
            add_debug_message("LLM Response is present")
            add_debug_message(f"Raw LLM Response: {raw_response}")
            
            # Use the parsing function
            llm_response = parse_llm_response(raw_response)
            
            # Debug the parsed LLM response
            add_debug_message("LLM Response parsed")
            add_debug_message(f"Parsed LLM Response: {llm_response}")
            
            if llm_response is not None:
                panel_number = llm_response.get("panel-number")
                # Set a default value if panel_number is None
                if panel_number is None:
                    panel_number = -1
                
                # Process Panel 0 (setup)
                if panel_number == 0:
                    # Extract values from the setup response
                    word_list = llm_response["setup"]["word-list"]
                    image_seed = llm_response["setup"]["image-seed"]
                    scenario_description_english = llm_response["setup"]["scenario-description-english"]
                    background_image_prompt_english = llm_response["setup"]["background-image-prompt-english"]
                    introduction_english = llm_response["setup"]["introduction-english"]
                    
                    # Print extracted values for debugging
                    add_debug_message(f"Word List: {word_list}")
                    add_debug_message(f"Image Seed: {image_seed}")
                    add_debug_message(f"Scenario Description (English): {scenario_description_english}")
                    add_debug_message(f"Background Image Prompt (English): {background_image_prompt_english}")
                    add_debug_message(f"Introduction (English): {introduction_english}")
                    
                    # Update session state with the extracted values
                    st.session_state.word_list = word_list
                    st.session_state.image_seed = image_seed
                    st.session_state.scenario_description_english = scenario_description_english
                    st.session_state.background_image_prompt_english = background_image_prompt_english
                    st.session_state.introduction_english = introduction_english
                    
                    # Convert word list to a newline-separated string for display in the text area
                    st.session_state.study_word_focus = "\n".join(word_list)
                    
                    # Update both seed and seed_value in session state
                    st.session_state.seed = image_seed
                    st.session_state.seed_value = image_seed
                    
                    # Process panel 0 (background generation and introduction)
                    st.session_state.scene_state['processing'] = True
                    process_panel(0, llm_response, background_description=background_image_prompt_english, seed=image_seed)
                    
                    # Automatically generate Panel 1 after Panel 0 is processed
                    st.session_state.generate_panel_1 = True
                
                # Process Panel 1+ (character and dialogue)
                elif panel_number >= 1:
                    # Extract the character image prompt and dialogue data
                    if "exchanges" in llm_response and len(llm_response["exchanges"]) > 0:
                        first_exchange = llm_response["exchanges"][0]
                        character_image_prompt = first_exchange.get("character-image-prompt-english", "")
                        
                        # Update the current position to match the panel number
                        st.session_state.scene_state["current_position"] = f"panel_{panel_number}"
                        
                        # Process the panel (character generation and dialogue)
                        st.session_state.scene_state['processing'] = True
                        process_panel(panel_number, llm_response, character_description=character_image_prompt, seed=st.session_state.seed)
                
                # Display the scenario description and introduction if they exist
                # Get values from session state if they're not defined in this code path
                scenario = st.session_state.get('scenario_description_english', '')
                intro = st.session_state.get('introduction_english', '')
                
                if scenario:
                    st.success(f"Scenario: {scenario}")
                if intro:
                    st.info(f"Introduction: {intro}")
                
                # Clear the LLM response to prevent reprocessing
                st.session_state.llm_response = None
                
                # Force a rerun to ensure all widgets update with the new values
                st.rerun()
            else:
                # Clear invalid response
                st.session_state.llm_response = None
        else:
            add_debug_message("No valid LLM response found in session state.")
    else:
        add_debug_message("No LLM response processing flag set.")
    
    # Create two columns - left for inputs, right for image display area
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
            # For AI dialogue, pass is_ai=True
            new_image, audio_content = process_dialogue(dialogue_line, current_image, current_offset, is_ai=True)
            
            # Update the combined image with the speech bubble
            st.session_state.scene_state['combined_image'] = new_image
            
            # Store the audio content
            audio_key = f"{current_pos}_{dialogue_index}"
            st.session_state.scene_state['dialogue_audio'][audio_key] = audio_content
            
            # Add a flag to indicate this is a new audio that should autoplay
            st.session_state.scene_state['new_audio_key'] = audio_key
            
            # Update the dialogue index and offset
            st.session_state.scene_state['current_dialogue_index'] += 1
            st.session_state.scene_state['dialogue_offsets'][current_pos] += 60
        
        # Reset the processing flags
        st.session_state.process_dialogue = False
        st.session_state.scene_state['processing'] = False
    
    with left_col:
        # Study word focus - display the word list from the LLM response
        study_word_focus = st.text_area(
            "Study word focus:",
            height=150,
            value=st.session_state.get('study_word_focus', ""),
            placeholder="Enter words or phrases to focus on..."
        )
        
        # Create two columns for the buttons
        button_col1, button_col2 = st.columns(2)
        
        # Button to load sample JLPT5 word list in the first column
        with button_col1:
            if st.button("Load Sample JLPT5 Word List"):
                try:
                    with open('seed_data/JLPT5_words.txt', 'r', encoding='utf-8') as file:
                        words = file.read().strip()
                        st.session_state.study_word_focus = words  # Store in session state
                        st.success("JLPT5 word list loaded successfully!")
                        st.rerun()  # Force a rerun to update the UI
                except Exception as e:
                    st.error(f"Error loading word list: {str(e)}")
        # Combined button to call Gemini for a new scenario in the second column
        with button_col2:
            if st.button("Call Gemini for a New Scenario"):
                try:
                    # Clear chat history
                    st.session_state.chat_history = []

                    # Clear dialogue lines array
                    if 'dialogue_lines_array' in st.session_state:
                        st.session_state.dialogue_lines_array = []
                    
                    # Save the current study words before resetting
                    current_study_words = study_word_focus.strip()
                    
                    # Reset scene state completely
                    default_state = {
                        'panels': {},  # Empty dictionary to store panels
                        'panel_count': 0,
                        'current_position': 'panel_0',  # Simplified position naming
                        'current_panel_number': 0,  # Track the current panel number (0 for setup, 1+ for content)
                        'panel_data': {},  # Store panel-specific data (character images, dialogue, etc.)
                        'current_stage': 'background',
                        'scene_background': None,
                        'background_image': None,
                        'character_image': None,
                        'combined_image': None,
                        'processing': False,
                        'dialogue_lines': [],
                        'current_dialogue_index': 0,
                        'dialogue_offsets': {},
                        'dialogue_audio': {}
                    }
                    st.session_state.scene_state = default_state
                    
                    # Clear all scenario-specific session variables
                    keys_to_clear = ['scenario_description_english', 'background_image_prompt_english', 
                                    'introduction_english', 'character_image_prompt_english', 
                                    'dialogue_japanese', 'word_list', 'seed_value']
                    
                    for key in keys_to_clear:
                        if key in st.session_state:
                            del st.session_state[key]
                    
                    # Clear any processing flags
                    if 'process_dialogue' in st.session_state:
                        del st.session_state['process_dialogue']
                    if 'generate_panel_1' in st.session_state:
                        del st.session_state['generate_panel_1']
                    
                    # Debug message to confirm reset
                    add_debug_message("Reset all session state variables for new scenario")
                    
                    # Read the LLM prompt template from file
                    with open('seed_data/LLM_prompt.md', 'r', encoding='utf-8') as file:
                        prompt_template = file.read().strip()
                    
                    # If the user has entered study words, include them in the prompt
                    if current_study_words:
                        add_debug_message(f"Using user-provided word list: {current_study_words}")
                        # Create a modified prompt that includes the user's word list
                        prompt_with_words = f"The user has provided the following study words:\n\n{current_study_words}\n\n{prompt_template}"
                        response = call_gemini_api(prompt_with_words)
                    else:
                        # If no words provided, let Gemini generate them as usual
                        add_debug_message("No user-provided word list, Gemini will generate one")
                        response = call_gemini_api(prompt_template)
                    
                    # Store the response in a new chat history
                    st.session_state.chat_history = [{"role": "assistant", "content": response}]
                    
                    # Store the response in llm_response and set the flag to process it
                    st.session_state.llm_response = response
                    st.session_state.process_llm_response = True
                    
                    # Process the response immediately to ensure widgets are populated on first click
                    # This helps bypass the need for a second click
                    raw_response = response
                    llm_response = parse_llm_response(raw_response)
                    
                    if llm_response is not None and llm_response.get("panel-number") == 0:
                        # Extract values from the setup response
                        word_list = llm_response["setup"]["word-list"]
                        image_seed = llm_response["setup"]["image-seed"]
                        scenario_description_english = llm_response["setup"]["scenario-description-english"]
                        background_image_prompt_english = llm_response["setup"]["background-image-prompt-english"]
                        introduction_english = llm_response["setup"]["introduction-english"]
                        
                        # Update session state with the extracted values
                        st.session_state.word_list = word_list
                        st.session_state.image_seed = image_seed
                        st.session_state.scenario_description_english = scenario_description_english
                        st.session_state.background_image_prompt_english = background_image_prompt_english
                        st.session_state.introduction_english = introduction_english
                        
                        # Convert word list to a newline-separated string for display in the text area
                        st.session_state.study_word_focus = "\n".join(word_list)
                        
                        # Update both seed and seed_value in session state
                        st.session_state.seed = image_seed
                        st.session_state.seed_value = image_seed
                        
                        # Immediately make a second API call to generate Panel 1
                        add_debug_message("Automatically generating Panel 1...")
                        
                        # Create a prompt for Panel 1
                        panel1_prompt = f"""NEW SESSION: Ignore all previous context. You are acting as a conversational partner in a Japanese language learning app. 
                        The user has responded to your previous dialogue line in our scenario: {scenario_description_english}.
                        
                        User's message: 
                        
                        Please respond with the next dialogue line in Japanese. Use natural conversational Japanese 
                        and try to incorporate one of these study words if possible: {', '.join(word_list[:5])}.
                        
                        Format your response as a JSON object with a dialogue-line-japanese field:
                        ```json
                        {{
                            "panel-number": 1,
                            "exchanges": [
                                {{
                                    "character-image-prompt-english": "[Detailed character description including appearance and emotion]",
                                    "dialogue-line-japanese": "[First dialogue line in Japanese using at least one study word]"
                                }}
                            ]
                        }}
                        ```"""
                        
                        # Call the API to generate Panel 1
                        panel1_response = call_gemini_api(panel1_prompt)
                        add_debug_message(f"Raw Panel 1 response: {panel1_response}")
                        
                        # Parse the Panel 1 response
                        panel1_data = parse_llm_response(panel1_response)
                        add_debug_message(f"Parsed Panel 1 data: {panel1_data}")
                        
                        if panel1_data is not None and panel1_data.get("panel-number") == 1:
                            # Store Panel 1 data in session state
                            add_debug_message(f"Successfully generated Panel 1: {panel1_data}")
                            
                            # Extract the character image prompt and dialogue line
                            if "exchanges" in panel1_data and len(panel1_data["exchanges"]) > 0:
                                first_exchange = panel1_data["exchanges"][0]
                                character_image_prompt = first_exchange.get("character-image-prompt-english", "")
                                
                                # Store in session state
                                st.session_state.character_image_prompt_english = character_image_prompt
                                
                                # Extract dialogue line from the first exchange
                                dialogue_line = first_exchange.get("dialogue-line-japanese", "")
                                
                                # Update the scene state to panel 1
                                st.session_state.scene_state["current_position"] = "panel_1"
                                st.session_state.scene_state["current_panel_number"] = 1
                                
                                # Process panel 1 using the new panel processing function
                                st.session_state.scene_state["processing"] = True
                                process_panel(1, panel1_data, character_description=character_image_prompt, seed=st.session_state.seed)
                        else:
                            add_debug_message("Failed to generate Panel 1 or response was invalid")
                
                    st.success("New scenario generated successfully!")
                    # Force a rerun to ensure all widgets update with the new values
                    st.rerun()
                except Exception as e:
                    st.error(f"Error generating scenario: {str(e)}")
        
        # Add a divider after the buttons
        st.divider()
        
        # Display scenario description in a new widget
        scenario_description = st.text_area(
            "Scenario:",
            height=80,
            value=st.session_state.get('scenario_description_english', ""),
            placeholder="Scenario description will appear here..."
        )
        
        # Display introduction in a new widget
        introduction = st.text_area(
            "Introduction:",
            height=100,
            value=st.session_state.get('introduction_english', ""),
            placeholder="Introduction will appear here..."
        )
        
        # Background description - display the background prompt from the LLM response
        background_description = st.text_area(
            "Background description:",
            height=100,
            value=st.session_state.get('background_image_prompt_english', ""),
            placeholder='e.g., "outside a theatre in a Japanese city"'
        )
        
        # This button has been moved up, right after the 'Load Sample JLPT5 Word List' button

        # Display the image seed as a simple text field (non-editable)
        st.write(f"Image seed: {st.session_state.get('seed_value', 0)}")
        
        # Add a divider before the character description
        st.divider()

        # Character description with predefined prompt
        character_description = st.text_area(
            "Character description:",
            height=100,
            value=st.session_state.get('character_image_prompt_english', ""),
            placeholder='e.g., "a Japanese woman smiling widely, short hair, square glasses, wearing dungarees"'
        )
        
        # This section was moved above, after the introduction field

        # Dialogue input with predefined prompt - accumulate conversation history
        # Initialize dialogue lines array if not already in session state
        if 'dialogue_lines_array' not in st.session_state:
            st.session_state.dialogue_lines_array = []
            
            # Initialize with the first AI response if available
            if st.session_state.get('dialogue_japanese', ""):
                initial_line = f"[AI]: {st.session_state.get('dialogue_japanese', '')}"
                st.session_state.dialogue_lines_array.append(initial_line)
        
        # Build dialogue history from the array
        dialogue_history = "\n\n".join(st.session_state.dialogue_lines_array)
        
        dialogue = st.text_area(
            "Dialogue:",
            height=200,  # Increased height to show more conversation
            value=dialogue_history,
            placeholder='e.g., "What a beautiful day!"'
        )
        
        # Process user input if dialogue has changed
        if dialogue != dialogue_history and dialogue.strip():
            # Extract the new line (assuming it's added at the end)
            new_lines = [line.strip() for line in dialogue.split('\n\n') if line.strip()]
            
            if len(new_lines) > len(st.session_state.dialogue_lines_array):
                # Get the newest line
                newest_line = new_lines[-1]
                
                # If it doesn't start with [AI] or [You], add the [You] prefix
                if not (newest_line.startswith('[AI]:') or newest_line.startswith('[You]:')):
                    newest_line = f"[You]: {newest_line}"
                
                # Add to array if it's new
                if newest_line not in st.session_state.dialogue_lines_array:
                    st.session_state.dialogue_lines_array.append(newest_line)
                    
                    # Add the dialogue to the panel if it's from the user
                    if newest_line.startswith('[You]:'):
                        user_text = newest_line[6:].strip()  # Remove the [You]: prefix
                        add_dialogue_to_panel(user_text, is_ai=False)
        
        # Generate buttons
        col3, col4 = st.columns(2)
        with col3:
            new_scene_button = st.button("New Scene")
        with col4:
            # Next panel button - always enabled
            next_panel_button = st.button("Next panel")
    
    # Show processing message in left column when generating
    if st.session_state.scene_state['processing']:
        st.info("Processing your prompts...")
    
    with right_col:
        # Add chat window at the top of the right column
        st.subheader("Gemini conversation...")
        
        # Create a container for the chat history with fixed height and scrolling
        chat_container = st.container(height=300, border=True)
        
        # Initialize chat history if it doesn't exist
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        
        # Display chat history in the container
        with chat_container:
            for message in st.session_state.chat_history:
                if message["role"] == "user":
                    st.markdown(f"**You:** {message['content']}")
                else:
                    st.markdown(f"**AI:** {message['content']}")
        
        # Create a form for the chat input to handle submission properly
        with st.form(key="chat_form", clear_on_submit=True):
            # Chat input field
            user_input = st.text_input("Type your message:")
            
            # Submit button
            submit_button = st.form_submit_button("Send")
            
            if submit_button and user_input:
                # Add user message to chat history
                st.session_state.chat_history.append({"role": "user", "content": user_input})
                
                # Add user dialogue to the panel with proper styling
                user_dialogue = f"[You]: {user_input}"
                add_dialogue_to_panel(user_dialogue, is_ai=False)
                
                # Create a prompt for the next dialogue line
                current_panel = st.session_state.scene_state.get('current_position', 'panel_1')
                current_dialogue_index = st.session_state.scene_state.get('current_dialogue_index', 0)
                
                # Get context from the current scenario
                scenario = st.session_state.get('scenario_description_english', "")
                word_list = st.session_state.get('word_list', [])
                
                # Create a prompt that instructs Gemini to continue the conversation
                dialogue_prompt = f"""You are acting as a conversational partner in a Japanese language learning app. 
                The user has responded to your previous dialogue line in our scenario: {scenario}.
                
                User's message: {user_input}
                
                Please respond with the next dialogue line in Japanese. Use natural conversational Japanese 
                and try to incorporate one of these study words if possible: {', '.join(word_list[:5])}.
                
                Format your response as a JSON object with a dialogue-line-japanese field:
                ```json
                {{
                    "panel-number": {current_panel.replace('panel_', '')},
                    "dialogue-line-japanese": "[Your Japanese response here]"
                }}
                ```"""
                
                # Call Gemini for a response
                response = call_gemini_api(dialogue_prompt)
                
                # Parse the JSON response to extract the dialogue line
                dialogue_data = parse_llm_response(response)
                dialogue_line = ""
                
                if dialogue_data and "dialogue-line-japanese" in dialogue_data:
                    dialogue_line = dialogue_data["dialogue-line-japanese"]
                    # Store the dialogue line in session state
                    st.session_state.dialogue_japanese = dialogue_line
                    
                    # Add AI dialogue to the panel with proper styling
                    ai_dialogue = f"[AI]: {dialogue_line}"
                    add_dialogue_to_panel(ai_dialogue, is_ai=True)
                
                # Add AI response to chat history with both the full response and the extracted dialogue line
                st.session_state.chat_history.append({
                    "role": "assistant", 
                    "content": response,
                    "dialogue_line": dialogue_line
                })
                
                # Check if the response contains JSON that should be processed
                if '{' in response and '}' in response:
                    # Store the response in llm_response and set the flag to process it
                    st.session_state.llm_response = response
                    st.session_state.process_llm_response = True
                
                # Force a rerun to update the chat history
                st.rerun()
        
        # Create layout for panels
        scene_container = st.container()
        with scene_container:
            # Create columns for panel display
            panel_left_col, panel_right_col = st.columns(2)
            
            # Display all completed panels in alternating left-right pattern
            panel_list = []
            for position, image in st.session_state.scene_state['panels'].items():
                if image is not None:
                    panel_list.append(image)
            
            # Display panels in alternating left-right pattern
            for i, panel in enumerate(panel_list):
                if i % 2 == 0:  # Even index (0, 2, 4...) - left column
                    with panel_left_col:
                        st.image(panel, use_container_width=True)
                else:  # Odd index (1, 3, 5...) - right column
                    with panel_right_col:
                        st.image(panel, use_container_width=True)
            
            # Display current working image in the appropriate column based on panel count
            current_image = None
            if st.session_state.scene_state['current_stage'] == 'background':
                current_image = st.session_state.scene_state['background_image']
            elif st.session_state.scene_state['current_stage'] in ['character', 'complete']:
                current_image = st.session_state.scene_state['combined_image']
            
            if current_image is not None:
                # Display in the next column based on panel count
                if len(panel_list) % 2 == 0:  # Even number of panels so far, use left column
                    with panel_left_col:
                        st.image(current_image, use_container_width=True)
                else:  # Odd number of panels so far, use right column
                    with panel_right_col:
                        st.image(current_image, use_container_width=True)
            
            # Play the most recent audio if available
            if st.session_state.scene_state['current_dialogue_index'] > 0:
                dialogue_index = st.session_state.scene_state['current_dialogue_index'] - 1
                current_pos = st.session_state.scene_state['current_position']
                audio_key = f"{current_pos}_{dialogue_index}"
                
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
                'panels': {},  # Empty dictionary to store panels
                'panel_count': 0,
                'current_position': 'panel_0',  # Simplified position naming
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
            # Save the current panel if it exists
            current_pos = st.session_state.scene_state['current_position']
            if st.session_state.scene_state.get('combined_image') is not None:
                st.session_state.scene_state['panels'][current_pos] = st.session_state.scene_state['combined_image']
            
            # Increment panel count
            st.session_state.scene_state['panel_count'] += 1
            
            # Create a new panel position
            next_pos = f"panel_{st.session_state.scene_state['panel_count']}"
            st.session_state.scene_state['current_position'] = next_pos
            st.session_state.scene_state['current_stage'] = 'character'  # Always start with character generation
            st.session_state.scene_state['background_image'] = st.session_state.scene_state['scene_background']  # Reuse background
            st.session_state.scene_state['character_image'] = None
            st.session_state.scene_state['combined_image'] = None
            st.session_state.scene_state['processing'] = True
            
            st.rerun()
        else:
            st.warning("Please enter both character and background descriptions.")
    
    # Process the current stage if we're in processing mode
    if st.session_state.scene_state['processing']:
        should_continue = process_current_stage(character_description, background_description, st.session_state.seed)
        
        # If processing was successful and we need to continue to the next stage, rerun the app
        if should_continue:
            st.rerun()

def process_panel(panel_number, panel_data, character_description=None, background_description=None, seed=None):
    """Process a specific panel based on its number and data.
    
    Args:
        panel_number: The panel number (0 for setup, 1+ for content panels)
        panel_data: Dictionary containing panel-specific data
        character_description: Character description for this panel (for panels 1+)
        background_description: Background description (for panel 0)
        seed: Random seed for image generation
        
    Returns:
        Boolean indicating if processing was successful
    """
    # Store the current panel number in session state
    st.session_state.scene_state['current_panel_number'] = panel_number
    
    # Panel 0 is for setup (background generation and introduction)
    if panel_number == 0:
        # Store panel data in the panel_data dictionary
        st.session_state.scene_state['panel_data'][panel_number] = panel_data
        
        # Process background generation
        background_success = process_background(background_description, seed)
        
        # Generate audio for introduction if available
        if 'introduction-english' in panel_data.get('setup', {}):
            introduction = panel_data['setup']['introduction-english']
            # TODO: Generate audio for introduction (future enhancement)
            
        return background_success
    
    # Panels 1+ are for character and dialogue
    else:
        # Store panel data in the panel_data dictionary
        if panel_number not in st.session_state.scene_state['panel_data']:
            st.session_state.scene_state['panel_data'][panel_number] = panel_data
        
        # Store character gender if available
        if "exchanges" in panel_data and len(panel_data["exchanges"]) > 0:
            first_exchange = panel_data["exchanges"][0]
            if 'character-gender' in first_exchange:
                st.session_state.scene_state['character_gender'] = first_exchange['character-gender']
        
        # Process character generation and overlay on background
        if character_description:
            character_success = process_character(character_description, seed)
            if not character_success:
                return False
        
        # Process dialogue if available
        # Check if dialogue is directly in panel_data
        if 'dialogue-line-japanese' in panel_data:
            dialogue_line = panel_data['dialogue-line-japanese']
            # Add AI dialogue to the panel with proper styling
            ai_dialogue = f"[AI]: {dialogue_line}"
            # Add to dialogue array
            if 'dialogue_lines_array' not in st.session_state:
                st.session_state.dialogue_lines_array = []
            st.session_state.dialogue_lines_array.append(ai_dialogue)
            add_dialogue_to_panel(dialogue_line, is_ai=True)
        # Check if dialogue is in the exchanges array
        elif 'exchanges' in panel_data and len(panel_data['exchanges']) > 0:
            first_exchange = panel_data['exchanges'][0]
            if 'dialogue-line-japanese' in first_exchange:
                dialogue_line = first_exchange['dialogue-line-japanese']
                # Add AI dialogue to the panel with proper styling
                ai_dialogue = f"[AI]: {dialogue_line}"
                # Add to dialogue array
                if 'dialogue_lines_array' not in st.session_state:
                    st.session_state.dialogue_lines_array = []
                st.session_state.dialogue_lines_array.append(ai_dialogue)
                add_dialogue_to_panel(dialogue_line, is_ai=True)
        
        return True

def process_background(background_description, seed):
    """Process background generation for a scene."""
    # Read the background style prompt suffix from file
    try:
        with open('seed_data/background_style_prompt_suffix.txt', 'r', encoding='utf-8') as file:
            background_style_suffix = file.read().strip()
        # Append the background style suffix to the background description
        enhanced_background_description = f"{background_description}. {background_style_suffix}"
        add_debug_message(f"Enhanced background prompt: {enhanced_background_description}")
    except Exception as e:
        add_debug_message(f"Error reading background style suffix: {str(e)}")
        enhanced_background_description = background_description
        
    # Generate background image only for the first panel in a scene
    background_image = generate_image(enhanced_background_description, steps=4, seed=seed)
    if background_image:
        # Apply Gaussian blur for combined view
        blurred_background = apply_gaussian_blur(background_image)
        
        # Store the background image for this scene
        st.session_state.scene_state['scene_background'] = blurred_background
        st.session_state.scene_state['background_image'] = blurred_background
        
        # Wait for a moment before generating character (reduced if image generation is disabled)
        if ENABLE_IMAGE_GENERATION:
            time.sleep(5)  # Reduced wait time
        else:
            time.sleep(1)  # Shorter wait time when image generation is disabled
        
        return True
    else:
        st.error("Failed to generate background image.")
        st.session_state.scene_state['processing'] = False
        return False

def process_character(character_description, seed):
    """Process character generation and overlay on background."""
    # Read the character style prompt suffix from file
    try:
        with open('seed_data/character_style_prompt_suffix.txt', 'r', encoding='utf-8') as file:
            character_style_suffix = file.read().strip()
        # Append the character style suffix to the character description
        enhanced_character_description = f"{character_description}. {character_style_suffix}"
        add_debug_message(f"Enhanced character prompt: {enhanced_character_description}")
    except Exception as e:
        add_debug_message(f"Error reading character style suffix: {str(e)}")
        enhanced_character_description = character_description
        
    # Generate character image
    character_image = generate_image(enhanced_character_description, steps=4, seed=seed)
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
        
        return True
    else:
        st.error("Failed to generate character image.")
        st.session_state.scene_state['processing'] = False
        return False

def process_current_stage(character_description, background_description, seed):
    """Process the current stage of scene generation."""
    current_stage = st.session_state.scene_state['current_stage']
    
    if current_stage == 'background':
        # Read the background style prompt suffix from file
        try:
            with open('seed_data/background_style_prompt_suffix.txt', 'r', encoding='utf-8') as file:
                background_style_suffix = file.read().strip()
            # Append the background style suffix to the background description
            enhanced_background_description = f"{background_description}. {background_style_suffix}"
            add_debug_message(f"Enhanced background prompt: {enhanced_background_description}")
        except Exception as e:
            add_debug_message(f"Error reading background style suffix: {str(e)}")
            enhanced_background_description = background_description
            
        # Generate background image only for the first panel in a scene
        background_image = generate_image(enhanced_background_description, steps=4, seed=seed)
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
            
            # Return True to indicate successful processing and that we should continue
            return True
        else:
            st.error("Failed to generate background image.")
            st.session_state.scene_state['processing'] = False
            return False
    
    elif current_stage == 'character':
        # Read the character style prompt suffix from file
        try:
            with open('seed_data/character_style_prompt_suffix.txt', 'r', encoding='utf-8') as file:
                character_style_suffix = file.read().strip()
            # Append the character style suffix to the character description
            enhanced_character_description = f"{character_description}. {character_style_suffix}"
            add_debug_message(f"Enhanced character prompt: {enhanced_character_description}")
        except Exception as e:
            add_debug_message(f"Error reading character style suffix: {str(e)}")
            enhanced_character_description = character_description
            
        # Generate character image
        character_image = generate_image(enhanced_character_description, steps=4, seed=seed)
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
            
            # Return True to indicate successful processing
            return True
        else:
            st.error("Failed to generate character image.")
            st.session_state.scene_state['processing'] = False
            return False

# Update the call_gemini_api function to accept a prompt parameter
def call_gemini_api(prompt, max_retries=3):
    """Call the Gemini API with the given prompt.
    
    Args:
        prompt: The prompt to send to the Gemini API.
        max_retries: Maximum number of retry attempts if valid JSON is not returned.
        
    Returns:
        The response text from the Gemini API.
    """
    try:
        # Load environment variables
        load_dotenv()
        
        # Configure Gemini
        api_key = os.getenv('GOOGLE_GEMINI_API_KEY')
        if not api_key:
            raise ValueError("GOOGLE_GEMINI_API_KEY not found in .env file")
        
        genai.configure(api_key=api_key)
        
        # Create a model instance
        model = genai.GenerativeModel('gemini-2.0-flash')
        
        # Try multiple times to get a valid JSON response
        for attempt in range(max_retries):
            # Generate a response based on the prompt
            response = model.generate_content(prompt)
            response_text = response.text
            
            # Check if the response contains valid JSON
            parsed_json = parse_llm_response(response_text)
            
            if parsed_json is not None:
                add_debug_message(f"Got valid JSON response on attempt {attempt + 1}")
                return response_text
            else:
                add_debug_message(f"Attempt {attempt + 1}/{max_retries} failed to produce valid JSON. Retrying...")
                
                # Add a stronger hint for the retry
                if attempt < max_retries - 1:  # Only add hint if we're going to retry
                    prompt = f"{prompt}\n\nIMPORTANT: Your previous response did not contain valid JSON. Please respond ONLY with a JSON object in the specified format. Do not include any explanatory text outside the JSON."
        
        # If we've exhausted all retries, return the last response
        add_debug_message("All retry attempts failed to produce valid JSON. Returning last response.")
        return response_text
        
    except Exception as e:
        print(f"Error calling Gemini API: {str(e)}")
        return f"Error generating response: {str(e)}"

if __name__ == "__main__":
    main() 