# 📄 RAG — Chat with your Document

A **Streamlit-based RAG (Retrieval-Augmented Generation) chatbot** that lets you **upload PDFs, DOCX, or TXT files** and ask questions directly from their content.  
It uses **OpenAI embeddings** for vector search and **Groq LLM (Llama 3.1 8B Instant)** for fast, accurate answers.  

## 🚀 Features

- Upload multiple **PDF / DOCX / TXT** files
- Build a **vector index** using OpenAI embeddings
- Query documents with **RAG-powered chat**
- Sources are cited with filenames & page numbers
- Adjustable **Top-K chunks**, **chunk size**, and **overlap**
- Interactive ChatGPT-like interface with `st.chat_message`
- Session memory with option to **clear chats and indexes**

## 🛠️ Tech Stack

- **Python 3.9+**
- **Streamlit** – Chat UI
- **LangChain** – RAG pipeline
- **Groq LLM** – Answer generation (`llama-3.1-8b-instant`)
- **OpenAI Embeddings** – `text-embedding-3-small`
- **FAISS** – Vector storage and retrieval
- **python-dotenv** – Environment variable management
- **PyPDFLoader / Docx2txtLoader / TextLoader** – Document loaders

## 📂 Project Structure

├── app.py # Main Streamlit app
├── Uploads/ # Uploaded documents (auto-created)
├── requirements.txt # Python dependencies
├── .env # API keys
└── README.md # Documentation

## ⚙️ Setup & Installation

1. **Clone this repository**
    ```bash
    git clone https://github.com/your-username/2.RAG-Document-QA.git
    cd 2.RAG-Document-QA
    ```

2. **Create a virtual environment**
    ```bash
    python -m venv venv
    source venv/bin/activate   # Mac/Linux
    venv\Scripts\activate      # Windows
    ```

3. **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```

4. **Configure environment variables**

    Create a `.env` file in the project root:
    ```env
    GROQ_API_KEY=your_groq_api_key_here
    LANGCHAIN_API_KEY=your_langchain_api_key_here   # optional, for LangSmith tracing
    ```

## ▶️ Usage (Local)

Run the app locally:

streamlit run app.py

- Open [http://localhost:8501] in your browser
- Enter your OpenAI API key in the sidebar (for embeddings)
- Upload one or more documents (PDF, DOCX, TXT)
- Click Build / Rebuild Index
- Ask questions in the chat box!

## 📜 License

This project is licensed under the MIT License.