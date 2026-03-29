# RAG Document Q&A

Ask questions about any PDF using Retrieval-Augmented Generation.

## Tech Stack
- LangChain — document chunking and pipeline
- HuggingFace sentence-transformers — text embeddings
- FAISS — vector similarity search
- Flan-T5 — answer generation (runs locally, no API key)
- Streamlit — web interface

## How it works
1. PDF is loaded and split into 500-character chunks
2. Each chunk is converted to a vector using all-MiniLM-L6-v2
3. User query is embedded with the same model
4. FAISS finds the top-3 most similar chunks
5. LLM generates an answer using those chunks as context

## Run locally
pip install -r requirements.txt
streamlit run app.py