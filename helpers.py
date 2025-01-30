import base64
import os
from glob import glob
import openai
from openai import OpenAI
from dotenv import load_dotenv
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory
import streamlit as st

# Load environment variables
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=api_key)
openai.api_key = api_key

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

def text_to_speech(input_text):
    response = client.audio.speech.create(
        model="tts-1",
        voice="nova",
        input=input_text
    )
    webm_file_path = "temp_audio_play.mp3"
    with open(webm_file_path, "wb") as f:
        response.stream_to_file(webm_file_path)
    return webm_file_path

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

def speech_to_text(audio_data, persona="neutral"):
    with open(audio_data, "rb") as audio_file:
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            response_format="text",
            file=audio_file
        )
    
    # Apply persona transformation
    return apply_persona(transcript, persona)

def apply_persona(response, persona):
    """Apply persona transformation to chatbot response."""
    persona_config = PERSONAS.get(persona, PERSONAS["neutral"])
    persona_response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": persona_config["system"]},
            {"role": "user", "content": f"{persona_config['instruction']}\n\n{response}"}
        ],
        temperature=0.7,
        max_tokens=300
    )
    return persona_response.choices[0].message.content

def base_model_chatbot(messages, persona="neutral"):
    system_message = [{"role": "system", "content": "You are an AI chatbot answering user queries."}]
    messages = system_message + messages

    response = client.chat.completions.create(
        model="gpt-3.5-turbo-1106",
        messages=messages
    )
    ai_response = response.choices[0].message.content

    return apply_persona(ai_response, persona)

class ConversationalRetrievalChain:
    """Class to manage the QA chain setup."""
    
    def __init__(self, model_name="gpt-3.5-turbo", temperature=0):
        self.model_name = model_name
        self.temperature = temperature
      
    def create_chain(self):
        model = ChatOpenAI(model_name=self.model_name, temperature=self.temperature)
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        vector_db = VectorDB('docs/')
        retriever = vector_db.create_vector_db().as_retriever(search_type="similarity", search_kwargs={"k": 2})

        return RetrievalQA.from_chain_type(llm=model, retriever=retriever, memory=memory)

def with_pdf_chatbot(messages, persona="neutral"):
    """Execute the QA system with persona transformation."""
    query = messages[-1]['content'].strip()
    qa_chain = ConversationalRetrievalChain().create_chain()
    result = qa_chain({"query": query})
    
    return apply_persona(result['result'], persona)
