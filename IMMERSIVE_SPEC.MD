# Immersive Japanese Learning Application Specification

## Overview

日常のスケッチ-EXP (Sketches of Everyday Life-EXP) is an experiment into a immersive Japanese language learning activity that creates interactive visual conversations to help users practice Japanese in context. The application generates manga-style panels with dialogue bubbles, allowing users to engage in natural Japanese conversations while practicing a user supplied vocabulary.

This formed part of investigations/experiments as part of :white_flower: https://genai.cloudprojectbootcamp.com/ :white_flower:

## Technical uncertainties
Focus areas for this prototype were
- integration of an LLM in a control role with an app - how consistent/reliable would giving over the driving role be in practice
-- LLM as server/app as client in nature
-- consistency of output from LLM - conversational wrap over the JSON we want

- using hosted LLMs for all main generation activites (free wherever possible)
-- it can often seem like it remains easier/faster/cheaper to use local code solutions over LLMs (although some open source solutions are ML/trained under the covers)

Note: in retrospect Streamlit seems a sub-optimal choice here, as it doesn't support any type of 'layers' and the control flow against LLM input challenging (at least for my knowledge-level/skill-set)

## Technologies

### Core Framework
- **Streamlit**: Web application framework for prototpying the user interface
- **Python 3.10+**: Core programming language

### Image Processing
- **PIL/Pillow**: Image manipulation for dialogue bubbles and overlays
- **OpenCV-Python**: Advanced image processing and background transparency

### AI/ML Integration
- **Google Gemini 2.0 Flash**: LLM for generating dialogue and controlling scenarios based on supplied vocabulary
- **FLUX.1-schnell-Free**: Image generation model (via Together API)
- **Google Text-to-Speech API**: Audio generation for Japanese dialogue

### Dependencies
- **numpy**: Numerical operations for image processing
- **requests**: HTTP requests for API communication
- **python-dotenv**: Environment variable management
- **together**: API wrapper for Together.ai services

## Features

1. **Scenario Generation**
   - Creates realistic Japanese conversation scenarios
   - Incorporates user-specified JLPT vocabulary words
   - Generates appropriate background and character descriptions

2. **Visual Scene Creation**
   - Generates background images based on scenario descriptions
   - Creates character images that match the dialogue context
   - Overlays characters on backgrounds with proper transparency

3. **Interactive Dialogue**
   - Displays dialogue in speech bubbles with distinct styling for user and AI
   - Supports natural conversation flow with the AI
   - Automatically positions and styles speech bubbles

4. **Audio Support**
   - Converts Japanese text to speech for pronunciation practice
   - Uses native Japanese voices for authentic pronunciation

5. **Learning Features**
   - Vocabulary integration with JLPT levels (N5-N1)
   - Context-based learning through realistic scenarios
   - Progressive difficulty based on JLPT level

## System Requirements

### API Keys Required
The application requires the following API keys to be set in a `.env` file:

```
FLUX_SCHNELL_FREE_API_KEY=your_flux_api_key
GOOGLE_GEMINI_API_KEY=your_gemini_api_key
GOOGLE_TTS_API_KEY=your_google_tts_api_key
```

### Prompt files

The application uses prompt files to generate/augment prompts for the LLMs. These files are in the /seed_data folder

- **LLM_PROMPT.md**: main prompt for Gemini 2.0 Flash
- **JLPT5_words.txt**: list of JLPT N5 words which can be loaded or you can supply your own.  Optional: if no words are supplied the LLM will create a list of 20 words at JLPT4. 
- **background_style_prompt_suffix.txt**: appended to LLM supplied description to provide a consistent visual style for the background - change to explore different styles and test consistency
- **character_style_prompt_suffix.txt**: appended to LLM supplied description to provide a consistent visual style for characters - change to explore different styles and test consistency - note also the image seed number which is randomise for each scenario but is consistent within it.


### Hardware Requirements
- Minimum 4GB RAM
- Internet connection for API access
- Modern web browser

## Architecture

The application follows a modular architecture:

1. **UI Layer** (Streamlit)
   - Handles user input and display
   - Manages application state and session variables

2. **LLM Integration Layer**
   - Communicates with Gemini API for dialogue generation
   - Parses and processes LLM responses

3. **Image Generation Layer**
   - Generates background and character images via FLUX API
   - Processes and combines images with proper transparency

4. **Dialogue Processing Layer**
   - Creates and positions speech bubbles
   - Manages dialogue flow and panel progression

5. **Audio Generation Layer**
   - Converts text to speech using Google TTS API
   - Handles audio playback in the application

## Development Flags

The application includes feature flags for development:
- `ENABLE_IMAGE_GENERATION`: Toggle image generation (background and character)
- `ENABLE_AUDIO_GENERATION`: Toggle audio generation for dialogue
- `DEBUG_MODE`: Enable detailed debug output

## Future Enhancements

Potential areas for future development:
- Exploring implementation of pure JSON output from LLM (Gemini) as it is still not always clean as supplied
- Resolving reliable narrative control from LLM
- Images are generated with FLUX.1 (Schnell)
-- free via together.ai - this is a free tier caps at 1 image per 10s - paid provision will remove this rate cap
-- Schnell does is not trained to supply alpha channels - look into alternative models which can output transparent images as flood-fill workaround only a prototype solution
-- other Flux.1 models also support input of reference images which could improve character consistency and unlock more variability in succesful prompts (styles, details) 
-- additionally other models support LORA but I did not find solutions here which are aimed at highly consistent 'concept art' character gen on alpha backgrounds which would seem to be the target for visual novel use
- Adding voice recognition for spoken user input
- Adding looping ambient soundtrack 
-- there does not seem to be much in the non-commercial service space as most models tuned for music creation not layering ambients based on prompts
-- something like https://github.com/remvze/moodist looks good and can be self-hosted but this experiment was all-LLM-all-the-time(tm) 
