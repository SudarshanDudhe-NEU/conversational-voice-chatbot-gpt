from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory
import os
from glob import glob
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


class VectorDB:
    def __init__(self, docs_directory: str = 'docs/'):
        self.docs_directory = docs_directory

    def create_vector_db(self):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        files = glob(os.path.join(self.docs_directory, "*.pdf"))
        loaders = [PyPDFLoader(pdf_file) for pdf_file in files]
        
        pdf_docs = []
        for loader in loaders:
            pdf_docs.extend(loader.load())
        chunks = text_splitter.split_documents(pdf_docs)
            
        return Chroma.from_documents(chunks, OpenAIEmbeddings())

class ConversationalRetrievalChain:
    def __init__(self, model_name="gpt-3.5-turbo", temperature=0):
        self.model_name = model_name
        self.temperature = temperature
      
    def create_chain(self):
        model = ChatOpenAI(model_name=self.model_name, temperature=self.temperature)
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        vector_db = VectorDB()
        retriever = vector_db.create_vector_db().as_retriever(
            search_type="similarity",
            search_kwargs={"k": 2}
        )
        return RetrievalQA.from_chain_type(
            llm=model,
            retriever=retriever,
            memory=memory
        )


def base_model_chatbot(messages):
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))  # Updated here
    
    system_message = [{
        "role": "system",
        "content": "You're a friendly AI assistant. Respond conversationally with brief, helpful answers."
    }]
    messages = system_message + messages
    
    response = client.chat.completions.create(
        model="gpt-3.5-turbo-1106",
        messages=messages,
        temperature=0.7
    )
    return response.choices[0].message.content

def with_pdf_chatbot(messages):
    query = messages[-1]['content'].strip()
    qa_chain = ConversationalRetrievalChain().create_chain()
    result = qa_chain({"query": query})
    return result['result']