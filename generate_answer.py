import os
import random 
import openai
from openai import OpenAI
from openai import AzureOpenAI
from persona_config import ACE_CONFIG
from dotenv import load_dotenv


# # Initialize Azure OpenAI client
# client = AzureOpenAI(
#     api_key=os.getenv("AZURE_OPENAI_KEY"),
#     api_version="2024-02-01",
#     azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
# )

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=api_key)
openai.api_key = api_key

def base_model_chatbot(messages):
    system_prompt = f"""
    You are {ACE_CONFIG['name']}, {ACE_CONFIG['role']} with {ACE_CONFIG['experience']}.
    Respond with:
    - Confidence: {ACE_CONFIG['traits']['confidence']}
    - Empathy: {ACE_CONFIG['traits']['empathy']}
    - Friendliness: {ACE_CONFIG['traits']['friendliness']}
    - Humor: {ACE_CONFIG['traits']['humor']}
    - Conciseness: {ACE_CONFIG['traits']['conciseness']}
    - Max {ACE_CONFIG['traits']['conciseness']} sentences
    - Use startup jargon appropriately
    """
    
    # response = client.chat.completions.create(
    #     engine="ace-startup-advisor",  # Your Azure deployment name
    #     messages=[{"role": "system", "content": system_prompt}] + messages,
    #     temperature=0.7,
    #     max_tokens=150,
    #     top_p=0.95
    # )
    
    system_message = [
        {"role": "system", "content": system_prompt}]
    messages = system_message + messages
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages
    )
    # return response.choices[0].message.content
    return _postprocess_azure_response(response.choices[0].message.content)

def _postprocess_azure_response(text):
    """Add conversational elements to Azure's response"""
    # # Add filler words
    # fillers = ACE_CONFIG['speech_patterns']['fillers']
    # filler = random.choice(fillers)
    # if len(text.split()) > 50 and not any(word in text.split()[15] for word in fillers):
    #     text = f"{filler} {text}"
    
    # Add check-in questions
    if "?" not in text:
        text += " How does that sound?"
    
    return text