# ResumeBot

A RAG-based Q&A tool for PDFs — upload a resume or document, ask
questions, get answers grounded in the actual content.

## How it works
1. Extracts text from the uploaded PDF
2. Chunks it with overlap for retrieval quality
3. Embeds chunks via Azure AI Inference and stores them in ChromaDB
4. On a question, retrieves the top matching chunks and passes them as
   context to GPT-4o for the final answer

## Tech Stack
Python · Streamlit · LangChain (text splitting) · ChromaDB · Azure AI
Inference (embeddings) · OpenAI (gpt-4o)

## Setup
```bash
pip install -r requirements.txt
# set GITHUB_TOKEN (or your model provider token) as an env var
streamlit run app.py
```

## Notes
Built as a focused exploration of a minimal RAG pipeline
