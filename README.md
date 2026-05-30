# RAG PDF Chatbot

A RAG-powered PDF Q&A web app built with LlamaIndex, ChromaDB, 
Gemini 2.5 Flash, and Streamlit.

## What it does
- Upload any PDF
- Ask questions about it
- Get grounded answers with source citations
- Powered by semantic search + LLM

## Tech Stack
- LLM: Gemini 2.5 Flash
- Embeddings: HuggingFace all-MiniLM-L6-v2
- RAG Framework: LlamaIndex
- Vector DB: ChromaDB
- UI: Streamlit

## Run Locally
pip install -r requirements.txt
streamlit run app.py

## Live Demo
https://rag-pdf-chatbot-yh267qtmr7iibpctharmyb.streamlit.app