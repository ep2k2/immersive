# Installation Guide for Immersive Japanese Learning Application

This guide will walk you through the process of setting up and running the "Everyday Sketch Experiment 日常のスケッチ-EXP for Immersive Japanese Learning" on your local machine.

## Prerequisites

- Python 3.10 or higher
- pip (Python package installer)
- Git (optional, for cloning the repository)

## Step 1: Clone or Download the Repository

If using Git:
```bash
git clone https://github.com/your-username/immersive.git
cd immersive
```

Or download and extract the ZIP file from the repository and navigate to the extracted directory.

## Step 2: Set Up a Virtual Environment (Recommended)

Creating a virtual environment is recommended to avoid conflicts with other Python packages:

```bash
# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

## Step 3: Install Dependencies

Install all required packages using pip:

```bash
pip install -r requirements.txt
```

This will install all necessary dependencies including:
- streamlit
- numpy
- pillow
- requests
- opencv-python
- python-dotenv
- together
- google-generativeai

## Step 4: Set Up API Keys

The application requires several API keys to function properly. Create a `.env` file in the root directory of the project with the following content:

```
FLUX_SCHNELL_FREE_API_KEY=your_flux_api_key
GOOGLE_GEMINI_API_KEY=your_gemini_api_key
GOOGLE_TTS_API_KEY=your_google_tts_api_key
```

Replace `your_flux_api_key`, `your_gemini_api_key`, and `your_google_tts_api_key` with your actual API keys:

1. **FLUX_SCHNELL_FREE_API_KEY**: Obtain from [Together.ai](https://www.together.ai/) (used for image generation)
2. **GOOGLE_GEMINI_API_KEY**: Get from [Google AI Studio](https://makersuite.google.com/app/apikey) (used for dialogue generation)
3. **GOOGLE_TTS_API_KEY**: Create in the [Google Cloud Console](https://console.cloud.google.com/) with Text-to-Speech API enabled (used for audio generation)

## Step 5: Testing API Connectivity

Before running the main application, it's recommended to test your API connections using the test scripts in the `test_LLM_connectivity` directory:

```bash
# Test Gemini API connection
python test_LLM_connectivity/test_gemini.py

# Test FLUX image generation
python test_LLM_connectivity/test_flux_schnell_via_together.py

# Test Google Text-to-Speech
python test_LLM_connectivity/test_google_tts.py
```

These tests will verify that your API keys are working correctly and that you can connect to the required services.

### Additional Test Files

The `test_LLM_connectivity` directory also contains tests for other LLM models that you may want to explore:

- `test_deepseek_r1_distill_llama_via_together.py`: Tests DeepSeek R1 Distill Llama model
- `test_llama_instruct_turbo_via_together.py`: Tests Llama Instruct Turbo model
- `test_llama_vision_free_via_together.py`: Tests Llama Vision Free model

These additional models are not currently used in the application but may be integrated in future versions.

## Step 6: Running the Application

Start the Streamlit application:

```bash
streamlit run app.py
```

This will launch the application in your default web browser. If it doesn't open automatically, the terminal will display a URL (typically http://localhost:8501) that you can open in your browser.

## Troubleshooting

### API Key Issues

If you encounter errors related to API keys:
1. Double-check that your `.env` file is in the correct location (root directory of the project)
2. Verify that your API keys are correct and have the necessary permissions
3. Run the specific test script for the failing API to get more detailed error information

### Image Generation Issues

If image generation is not working:
1. Check the `ENABLE_IMAGE_GENERATION` flag in `app.py` is set to `True`
2. Verify your FLUX API key is valid by running the test script
3. Check your internet connection as image generation requires online access

### Audio Generation Issues

If audio generation is not working:
1. Check the `ENABLE_AUDIO_GENERATION` flag in `app.py` is set to `True`
2. Verify your Google TTS API key is valid by running the test script
3. Ensure the Google Cloud Text-to-Speech API is enabled in your Google Cloud project

## Development Mode

For development purposes, you can modify the feature flags at the top of `app.py`:

```python
# Feature flags for development
ENABLE_IMAGE_GENERATION = True  # Set to False to skip image generation
ENABLE_AUDIO_GENERATION = False  # Set to False to skip audio generation
DEBUG_MODE = True  # Set to True to enable debug output
```

Streamlit automatically hot-reloads when code changes are detected, so there's no need to restart the application after making changes to the code or prompt files.
