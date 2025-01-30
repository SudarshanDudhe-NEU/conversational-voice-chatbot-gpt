import streamlit as st
import hmac
import os
from helpers import text_to_speech, autoplay_audio, speech_to_text, base_model_chatbot, with_pdf_chatbot
from audio_recorder_streamlit import audio_recorder
from streamlit_float import *

# Persona configurations
PERSONAS = {
    "neutral": {
        "system": "You are a helpful assistant.",
        "instruction": "Return the text exactly as received without modification."
    },
    "cheerful": {
        "system": "You are an exceptionally cheerful assistant.",
        "instruction": "Make this text sound upbeat and positive."
    },
    "formal": {
        "system": "You are a formal business assistant.",
        "instruction": "Convert this text to professional business language."
    },
    "pirate": {
        "system": "You are a pirate from the Caribbean.",
        "instruction": "Convert this text into pirate slang."
    }
}

def main(answer_mode: str):
    # Initialize floating UI
    float_init()

    # Initialize session state
    def initialize_session_state():
        if "messages" not in st.session_state:
            st.session_state.messages = [
                {"role": "assistant", "content": "Hi! How may I assist you today?"}
            ]
        if "selected_persona" not in st.session_state:
            st.session_state.selected_persona = "neutral"  # Default persona

    initialize_session_state()

    # App Title
    st.title("OpenAI Conversational Chatbot 🤖")

    # **Persona Selector**
    st.session_state.selected_persona = st.selectbox(
        "Select Conversation Style:",
        list(PERSONAS.keys()),
        index=list(PERSONAS.keys()).index(st.session_state.selected_persona),
    )

    # Create footer container for the microphone
    footer_container = st.container()
    with footer_container:
        audio_bytes = audio_recorder()

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # Process audio input
    if audio_bytes:
        with st.spinner("🎤 Transcribing..."):
            webm_file_path = "temp_audio.mp3"
            with open(webm_file_path, "wb") as f:
                f.write(audio_bytes)

            transcript = speech_to_text(webm_file_path, st.session_state.selected_persona)  # Apply persona-based transcription
            if transcript:
                st.session_state.messages.append({"role": "user", "content": transcript})
                with st.chat_message("user"):
                    st.write(transcript)
                os.remove(webm_file_path)

    # Generate response
    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("🤔 Thinking..."):
                if answer_mode == 'base_model':
                    final_response = base_model_chatbot(st.session_state.messages, st.session_state.selected_persona)
                elif answer_mode == 'pdf_chat':
                    final_response = with_pdf_chatbot(st.session_state.messages, st.session_state.selected_persona)

            # Convert response to speech
            with st.spinner("🎙️ Generating audio response..."):
                audio_file = text_to_speech(final_response)
                autoplay_audio(audio_file)

            st.write(final_response)
            st.session_state.messages.append({"role": "assistant", "content": final_response})
            os.remove(audio_file)

if __name__ == "__main__":
    main(answer_mode='base_model')  # Default to base model
