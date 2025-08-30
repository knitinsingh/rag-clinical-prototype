# Minimal RAG Prototype (CLI)

This project is a **hands-on Retrieval-Augmented Generation (RAG) prototype** designed to show how physicians (or clinical staff) could ask free-text questions about patient records and receive context-aware answers.  

The prototype:
- Loads a small set of clinical text files (e.g., MIMIC-style notes).
- Splits documents into chunks for better retrieval.
- Stores embeddings in a local **Chroma** vector store.
- Retrieves the most relevant context for a user’s query.
- Uses a **public LLM (OpenAI)** or a **local LLM (Ollama)** to generate an answer strictly from the retrieved context.

---

## 🎯 Objective
The goal is to demonstrate how RAG can help a physician **quickly find clinical facts** from large, unstructured records (like discharge summaries or radiology reports).  

For example:  
> *“Which lung had the cavitary lesions on the chest CT?”*  
The system retrieves the relevant CT report snippet and responds:  
**“The left lung had the cavitary lesions on the chest CT.”**

---

## 📂 Data Source
We use sample text inspired by the **MIMIC clinical dataset** (Medical Information Mart for Intensive Care).  
- MIMIC is a large, de-identified clinical database widely used in research.  
- Here, only a few **dummy `.txt` notes** are included (radiology reports, discharge summaries).  
- The pipeline is generic: you can replace these with any text files (clinical or non-clinical).

---

## 🛠️ Stack
- **Chunking**: Simple sliding window  
- **Vector store**: Chroma (cosine similarity)  
- **Embeddings**: Ollama `nomic-embed-text`  
- **LLM for answering**:  
  - Default: OpenAI (`gpt-4o-mini`) – public API  
  - Optional: Ollama local model (`llama3.2`)  

---

## 🔄 Architecture

```mermaid
flowchart LR
  A[Text files] --> B[Chunking]
  B --> C[Embeddings]
  C --> D[Chroma vector store]
  E[User question] --> D
  D --> G[Top-k relevant chunks]
  G --> H[LLM answer]
  H --> I[Final response]


## Setup 
### 1) Python environment
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

### 2) Add at least 3 text files
mkdir -p data
### copy your .txt files into ./data

### 3) Start Ollama (for embeddings)
ollama serve
ollama pull nomic-embed-text

### 4) OpenAI key (public LLM)
export OPENAI_API_KEY="sk-xxxxxxxx"

### Build the index (embeds & stores chunks)
python rag_cli.py --rebuild

### Ask questions (OpenAI for answers)
python rag_cli.py --provider openai

### Or fully local answering (optional)
python rag_cli.py --provider ollama --model llama3.2

### Build the index (embeds & stores chunks)
python rag_cli.py --rebuild

### Ask questions (OpenAI for answers)
python rag_cli.py --provider openai

### Or fully local answering (optional)
python rag_cli.py --provider ollama --model llama3.2

