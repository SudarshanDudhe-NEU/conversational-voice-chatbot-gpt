import os
import base64
import streamlit as st
import azure.cognitiveservices.speech as speechsdk
from persona_config import ACE_CONFIG

import openai
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# client = OpenAI(api_key=api_key)
client = OpenAI()
openai.api_key = api_key

import os
import azure.cognitiveservices.speech as speechsdk

# def recognize_from_microphone():
#     # This example requires environment variables named "SPEECH_KEY" and "SPEECH_REGION"
#     speech_config = speechsdk.SpeechConfig(subscription=os.getenv('AZURE_SPEECH_KEY'), region=os.getenv('AZURE_REGION'))
#     speech_config.speech_recognition_language="en-US"

#     audio_config = speechsdk.audio.AudioConfig(use_default_microphone=True)
#     speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)

#     print("Speak into your microphone.")
#     speech_recognition_result = speech_recognizer.recognize_once_async().get()

#     if speech_recognition_result.reason == speechsdk.ResultReason.RecognizedSpeech:
#         print("Recognized: {}".format(speech_recognition_result.text))
#     elif speech_recognition_result.reason == speechsdk.ResultReason.NoMatch:
#         print("No speech could be recognized: {}".format(speech_recognition_result.no_match_details))
#     elif speech_recognition_result.reason == speechsdk.ResultReason.Canceled:
#         cancellation_details = speech_recognition_result.cancellation_details
#         print("Speech Recognition canceled: {}".format(cancellation_details.reason))
#         if cancellation_details.reason == speechsdk.CancellationReason.Error:
#             print("Error details: {}".format(cancellation_details.error_details))
#             print("Did you set the speech resource key and region values?")


def speech_to_text(audio_data):
    
    with open(audio_data, "rb") as audio_file:
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            response_format="text",
            file=audio_file
        )
    return transcript


def text_to_speech(input_text):
    # Azure Speech Configuration
    speech_key = os.getenv("AZURE_SPEECH_KEY")
    service_region = os.getenv("AZURE_REGION")
    
    speech_config = speechsdk.SpeechConfig(
        subscription=speech_key, 
        region=service_region
    )
    
    # Configure Ace's voice
    speech_config.speech_synthesis_voice_name = "en-US-AvaMultilingualNeural"
    # speech_config.set_property(
    #     "SpeechServiceConnection_SynthVoiceFormat", 
    #     "audio-16khz-32kbitrate-mono-mp3"
    # )
    speech_config.set_speech_synthesis_output_format(
        speechsdk.SpeechSynthesisOutputFormat.Audio16Khz32KBitRateMonoMp3
    )


    # Build SSML with persona adjustments
    ssml_text = f"""
    <speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" 
           xmlns:mstts="https://www.w3.org/2001/mstts" xml:lang="en-US">
        <voice name="{speech_config.speech_synthesis_voice_name}">
            <prosody rate="{ACE_CONFIG['traits']['pace']}" pitch="+1st">
                {_apply_speech_patterns(input_text)}
            </prosody>
        </voice>
    </speak>
    """
    
    # Synthesize speech
    synthesizer = speechsdk.SpeechSynthesizer(
        speech_config=speech_config, 
        audio_config=None
    )
    result = synthesizer.speak_ssml_async(ssml_text).get()
    
    # Save audio file
    webm_file_path = "temp_audio_play.mp3"
    with open(webm_file_path, "wb") as f:
        f.write(result.audio_data)
    
    return webm_file_path

def _apply_speech_patterns(text):
    """Add SSML markup for Ace's speech patterns"""
    # Add pauses
    text = text.replace("...",ACE_CONFIG['speech_patterns']['pauses']['long'])
    text = text.replace(",", ACE_CONFIG['speech_patterns']['pauses']['short'])
    
    # Add dynamic styles
    if any(keyword in text.lower() for keyword in ACE_CONFIG['speech_patterns'] ['emotional_tone']['cheerful']):
        return f"<mstts:express-as style='cheerful'>{text}</mstts:express-as>"
    elif any(keyword in text.lower() for keyword in ACE_CONFIG['speech_patterns']['emotional_tone']['empathetic']):
        return f"<mstts:express-as style='empathetic'>{text}</mstts:express-as>"
    
    return text



def autoplay_audio(file_path: str):
    with open(file_path, "rb") as f:
        data = f.read()
    b64 = base64.b64encode(data).decode("utf-8")
    md = f"""
    <audio autoplay>
    <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
    </audio>
    """
    st.markdown(md, unsafe_allow_html=True)
    # recognize_from_microphone()

