import streamlit as st
from PyPDF2 import PdfReader
import pandas as pd
import base64
from datetime import datetime

# Langchain Imports
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.chains.question_answering import load_qa_chain
from langchain_core.prompts import PromptTemplate

from datetime import datetime

st.set_page_config(page_title="Chat with multiple PDFs", page_icon=":books:", layout="wide")

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            content = page.extract_text()
            if content:
                text += content
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return text_splitter.split_text(text)

def get_vector_store(text_chunks, api_key):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")
    return vector_store

def get_conversational_chain(api_key):
    prompt_template = """
    Answer the question as detailed as possible from the provided context. If the answer is not in
    the provided context, just say "Answer is not available in the context".\n\n
    Context:\n {context}?\n
    Question: \n{question}\n
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3, google_api_key=api_key)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

HTML_STYLE = """
<style>
    .chat-container { margin-bottom: 20px; }
    .chat-message {
        padding: 1rem; border-radius: 0.5rem; margin-bottom: 1rem; display: flex; align-items: flex-start;
    }
    .chat-message.user { background-color: #2b313e; }
    .chat-message.bot { background-color: #475063; }
    .chat-message .avatar { width: 50px; flex-shrink: 0; }
    .chat-message .avatar img {
        width: 40px; height: 40px; border-radius: 50%; object-fit: cover;
    }
    .chat-message .message { flex-grow: 1; margin-left: 15px; color: #fff; }
</style>
"""

def main():
    st.markdown(HTML_STYLE, unsafe_allow_html=True)
    st.header("Chat with multiple PDFs :books:")

    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []

    # Sidebar
    st.sidebar.title("Configuration")
    model_name = st.sidebar.radio("Select Model:", ("Google AI",))
    api_key = st.sidebar.text_input("Enter Google API Key:", type="password")
    
    with st.sidebar:
        st.divider()
        pdf_docs = st.file_uploader("Upload PDFs", accept_multiple_files=True)
        if st.button("Submit & Process"):
            if pdf_docs and api_key:
                with st.spinner("Processing..."):
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks, api_key)
                    st.success("Analysis Complete!")
            else:
                st.error("Please upload PDFs and enter API Key.")

        if st.sidebar.button("Clear Chat"):
            st.session_state.conversation_history = []
            st.rerun()

    # Chat Input
    user_question = st.chat_input("Ask a question about your documents...")

    if user_question:
        if not api_key:
            st.error("Please provide an API Key in the sidebar.")
        else:
            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
            new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
            docs = new_db.similarity_search(user_question)
            
            chain = get_conversational_chain(api_key)
            response = chain.invoke({"input_documents": docs, "question": user_question}, return_only_outputs=True)
            answer = response["output_text"]

            # Store in history
            st.session_state.conversation_history.append({
                "question": user_question, 
                "answer": answer,
                "timestamp": datetime.now().strftime('%H:%M:%S')
            })

            st.snow()

    # Display History (Corrected Logic)
    for chat in st.session_state.conversation_history:
        # User Message
        st.markdown(f"""
        <div class="chat-message user">
            <div class="avatar"><img src="https://i.ibb.co/CKpTnWr/user-icon-2048x2048-ihoxz4vq.png"></div>
            <div class="message">{chat['question']}</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Bot Message
        st.markdown(f"""
        <div class="chat-message bot">
            <div class="avatar"><img src="https://i.ibb.co/wNmYHsx/langchain-logo.webp"></div>
            <div class="message">{chat['answer']}</div>
        </div>
        """, unsafe_allow_html=True)

    # Download Section
    if st.session_state.conversation_history:
        df = pd.DataFrame(st.session_state.conversation_history)
        csv = df.to_csv(index=False).encode('utf-8')
        st.sidebar.download_button("Download History (CSV)", data=csv, file_name="chat_history.csv", mime="text/csv")

if __name__ == "__main__":
    main()