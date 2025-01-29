🗣️ OpenAI Conversational Voice Chatbot 🤖🎙️

An AI-powered voice-enabled chatbot built using OpenAI’s GPT, LangChain, Streamlit, and ChromaDB. This application enables real-time conversation using speech-to-text (STT) and text-to-speech (TTS), allowing users to interact naturally with an AI assistant.

🚀 Features

✅ Conversational AI – Uses OpenAI’s gpt-3.5-turbo or gpt-4 for intelligent responses.
✅ Voice Input & Output – Integrates OpenAI’s Whisper for speech-to-text and TTS for natural voice responses.
✅ Interactive UI – Built with Streamlit, offering a clean and user-friendly chatbot experience.
✅ Document-Based Q&A – Supports PDF-based knowledge retrieval using LangChain and ChromaDB.
✅ Persistent Memory – Maintains conversation history for a smooth chat experience.
✅ Scalable & Deployable – Can be deployed on Streamlit Cloud, Hugging Face Spaces, or AWS/GCP.

🛠️ Tech Stack
	•	Backend: OpenAI GPT, LangChain
	•	Speech Processing: Whisper (STT), OpenAI TTS
	•	Vector Database: ChromaDB
	•	Frontend: Streamlit
	•	Environment Management: Python, Dotenv

📦 Installation & Setup

1️⃣ Clone the repository

git clone https://github.com/your-username/openai-voice-chatbot.git
cd openai-voice-chatbot

2️⃣ Create a virtual environment

python -m venv .venv
source .venv/bin/activate  # macOS/Linux

3️⃣ Install dependencies

pip install -r requirements.txt

4️⃣ Set up OpenAI API key
Create a .env file and add:

OPENAI_API_KEY=your_api_key_here

5️⃣ Run the chatbot

streamlit run app.py

📌 Future Improvements

🔹 Support for multiple voices in TTS
🔹 Integration with additional document formats (Word, TXT)
🔹 Enhanced chatbot memory using vector search
🔹 Deployment guide for cloud platforms

📄 License

This project is open-source and available under the MIT License.

🌟 Contributions

Contributions, bug reports, and feature suggestions are welcome! Feel free to fork this repository and submit a pull request.

🚀 Happy Chatting! 🎙️💬
