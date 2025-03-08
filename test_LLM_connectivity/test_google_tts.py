import os
import requests
import base64
from dotenv import load_dotenv
from io import BytesIO
import sys
import time

def test_google_tts_connectivity():
    """Test connectivity to Google's Text-to-Speech API."""
    print("Testing Google Text-to-Speech API connectivity...")
    
    # Load environment variables
    load_dotenv()
    api_key = os.getenv('GOOGLE_TTS_API_KEY')
    
    if not api_key:
        print("❌ ERROR: GOOGLE_TTS_API_KEY not found in .env file")
        print("Please add your Google TTS API key to the .env file as GOOGLE_TTS_API_KEY=your_api_key")
        return False
    
    # Test text (both English and Japanese)
    test_texts = [
        "Hello, this is a test of the Google Text-to-Speech API.",
        "こんにちは、これはGoogle Text-to-Speech APIのテストです。"
    ]
    
    url = "https://texttospeech.googleapis.com/v1/text:synthesize"
    
    for i, text in enumerate(test_texts):
        language_code = "en-US" if i == 0 else "ja-JP"
        voice_name = "en-US-Standard-D" if i == 0 else "ja-JP-Standard-D"
        
        print(f"\nTesting with {language_code} text: '{text}'")
        
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
                "languageCode": language_code,
                "name": voice_name
            }
        }
        
        headers = {
            "Content-Type": "application/json",
            "X-Goog-Api-Key": api_key
        }
        
        try:
            start_time = time.time()
            response = requests.post(url, json=payload, headers=headers)
            end_time = time.time()
            
            if response.status_code == 200:
                response_data = response.json()
                
                if 'audioContent' in response_data:
                    audio_content = base64.b64decode(response_data['audioContent'])
                    audio_size = len(audio_content)
                    
                    print(f"✅ SUCCESS: Generated {audio_size} bytes of audio in {end_time - start_time:.2f} seconds")
                    
                    # Save the audio file for verification
                    output_file = f"test_output_{language_code}.wav"
                    with open(output_file, "wb") as f:
                        f.write(audio_content)
                    print(f"   Audio saved to {output_file}")
                else:
                    print(f"❌ ERROR: Unexpected response structure: {response_data}")
                    return False
            else:
                print(f"❌ ERROR: API request failed with status code {response.status_code}")
                print(f"Response: {response.text}")
                return False
                
        except Exception as e:
            print(f"❌ ERROR: Exception occurred: {str(e)}")
            return False
    
    print("\n✅ All Google TTS API tests passed successfully!")
    return True

if __name__ == "__main__":
    success = test_google_tts_connectivity()
    sys.exit(0 if success else 1) 