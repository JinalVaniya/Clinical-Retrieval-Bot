# Clinical Retrieval Assistant

A Retrieval-Augmented Generation (RAG) based healthcare chatbot that enables users to query medical PDF documents using natural language.

The system retrieves relevant information from indexed medical documents and generates context-grounded responses using **Groq's Llama 3.1 8B Instant** model.

---

## Features

* Medical PDF document ingestion
* Semantic search using embeddings
* FAISS vector database
* Retrieval-Augmented Generation (RAG)
* Groq-hosted LLM integration
* Source-backed responses
* Streamlit chat interface
* Clinician-focused prompt engineering

---

## Architecture

```text
Medical PDFs
      │
      ▼
PyMuPDFLoader
      │
      ▼
Text Chunking
      │
      ▼
HuggingFace Embeddings
      │
      ▼
FAISS Vector Store
      │
      ▼
Retriever
      │
      ▼
Groq LLM
      │
      ▼
Generated Response + Sources
```

---

## Tech Stack

* Python
* LangChain
* LangChain Groq
* LangChain HuggingFace
* FAISS
* HuggingFace Embeddings
* Streamlit
* PyMuPDF
* python-dotenv

---

## Project Structure

```text
project/
│
├── Data/
├── vectorstore/
│   └── db_faiss/
│
├── extract_vector_data.py
├── LLM_connection.py
├── app.py
├── pyproject.toml
├── .env.example
└── README.md
```

---

## Environment Variables

Create a `.env` file:

```env
GROQ_API_KEY=YOUR_GROQ_API_KEY
HF_TOKEN=YOUR_HUGGINGFACE_TOKEN   
```

### API Keys

Groq:
https://console.groq.com/keys

Hugging Face:
https://huggingface.co/settings/tokens

---

## Build Vector Database

Place medical PDF documents inside the `Data/` folder.

Run:

```bash
python extract_vector_data.py
```

This will:

* Load PDF documents
* Create text chunks
* Generate embeddings
* Build the FAISS vector database

---

## Run Chatbot

### Command Line Version

```bash
python LLM_connection.py
```

### Streamlit Interface

```bash
streamlit run app.py
```

---

## Example Query

```text
What are the treatment options for cancer?
```

The system:

1. Retrieves relevant document chunks
2. Passes retrieved context to the LLM
3. Generates a grounded response
4. Displays supporting source documents

---

## Future Improvements

* Hybrid Search (BM25 + Vector Search)
* Clinical Guideline Integration
* Confidence Scoring
* FHIR Integration
* Citation Ranking
* Multi-document Collections
