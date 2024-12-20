import os
import sys
import azure.cognitiveservices.speech as speechsdk
from flask import Flask, request, send_from_directory, render_template, flash, redirect, url_for

app = Flask(__name__)
app.secret_key = 'c897d534a33b4dd7a31e73026200226b'  # Required for flashing messages

# Azure Speech Configuration - IMPORTANT: REPLACE WITH YOUR ACTUAL CREDENTIALS
SPEECH_KEY = "18f978cca70246309254196a93ce34b4"  # Recommended: Use environment variable
SERVICE_REGION = "eastus"
ENDPOINT_ID = "18ef11c9-f8f5-4168-aba4-bd0db4e0a95b"
VOICE_NAME = "drdavidNeural"

# Ensure a clean, absolute path for static files
STATIC_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static')
os.makedirs(STATIC_FOLDER, exist_ok=True)


def synthesize_speech(text):
    """Synthesize speech using Azure Custom Voice"""
    # Validate input parameters
    if not SPEECH_KEY:
        raise ValueError("Azure Speech Key is not set. Please set the AZURE_SPEECH_KEY environment variable.")
    
    try:
        speech_config = speechsdk.SpeechConfig(subscription=SPEECH_KEY, region=SERVICE_REGION)
        speech_config.endpoint_id = ENDPOINT_ID
        speech_config.speech_synthesis_voice_name = VOICE_NAME
        speech_config.set_speech_synthesis_output_format(
            speechsdk.SpeechSynthesisOutputFormat.Audio24Khz160KBitRateMonoMp3
        )

        # Use a consistent, absolute path
        output_filename = os.path.join(STATIC_FOLDER, 'synthesized_audio.mp3')

        file_config = speechsdk.audio.AudioOutputConfig(filename=output_filename)
        speech_synthesizer = speechsdk.SpeechSynthesizer(
            speech_config=speech_config, 
            audio_config=file_config
        )

        result = speech_synthesizer.speak_text_async(text).get()

        if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
            # Return just the filename, not the full path
            return 'synthesized_audio.mp3'
        elif result.reason == speechsdk.ResultReason.Canceled:
            cancellation_details = result.cancellation_details
            error_message = f"Speech synthesis canceled: {cancellation_details.reason}"
            if cancellation_details.reason == speechsdk.CancellationReason.Error:
                error_message += f" Error details: {cancellation_details.error_details}"
            raise Exception(error_message)
    
    except Exception as e:
        print(f"Error in speech synthesis: {e}")
        raise

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        text = request.form.get('text', '').strip()
        if not text:
            return {"error": "Please enter some text to synthesize."}, 400
        
        try:
            audio_file = synthesize_speech(text)
            return {"audio_file": audio_file}, 200
        except ValueError as ve:
            return {"error": str(ve)}, 400
        except Exception as e:
            return {"error": f"An error occurred during speech synthesis: {e}"}, 500
    
    return render_template('hello.html')


@app.route('/audio/<filename>')
def serve_audio(filename):
    return send_from_directory(STATIC_FOLDER, filename, mimetype='audio/mpeg')

if __name__ == '__main__':
    # Validate Speech Key at startup
    if not SPEECH_KEY:
        print("ERROR: Azure Speech Key is not set. Please set the AZURE_SPEECH_KEY environment variable.")
        sys.exit(1)
    
    app.run(debug=True)
