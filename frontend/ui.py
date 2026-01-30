import streamlit as st
import requests

API_URL = "http://localhost:8000/chat"

st.set_page_config(page_title="Agentic AI RAG Chatbot")

st.title("🤖 Agentic AI RAG Chatbot (FAISS + Ollama)")
query = st.text_input("Ask a question from the Agentic AI eBook")

if st.button("Ask") and query:
    response = requests.post(API_URL, params={"query": query})
    data = response.json()

    st.success(data["final_answer"])
    st.write("Confidence:", data["confidence"])

    with st.expander("Retrieved Context"):
        for ctx in data["retrieved_context"]:
            st.write(ctx)
