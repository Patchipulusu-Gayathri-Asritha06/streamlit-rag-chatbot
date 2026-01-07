# Streamlit-Gemini-RAG-Chatbot

A **Retrieval-Augmented Generation (RAG) based chatbot** that allows users to query and interact with **multiple PDF documents** using natural language. This project leverages **LangChain**, **Google Gemini**, and **FAISS vector database** to provide accurate, context-aware answers from unstructured document data.

---

## Features

- **PDF Upload & Processing:** Extracts text from multiple PDFs and splits it into semantic chunks for better context handling.
- **Semantic Search with FAISS:** Generates embeddings using **Google Generative AI Embeddings** and stores them in a FAISS vector database for fast similarity search.
- **Context-Aware QA:** Uses **LangChain QA chains** and **Google Gemini** to answer questions based strictly on document content.
- **Interactive Streamlit Interface:** Chat interface with conversation history, timestamps, and downloadable CSV logs.
- **Modular & Scalable:** Easily extendable to add new AI models, documents, or features.

---

## Tech Stack

- **Python 3.10+**
- **Streamlit** – Web interface  
- **LangChain** – RAG & QA chains  
- **Google Gemini API** – AI model for response generation  
- **FAISS** – Vector database for semantic search  
- **PyPDF2** – PDF text extraction  
- **Pandas** – Data handling and CSV export  

---
   
git clone https://github.com/<your-username>/streamlit-gemini-rag-chatbot.git
cd streamlit-gemini-rag-chatbot
