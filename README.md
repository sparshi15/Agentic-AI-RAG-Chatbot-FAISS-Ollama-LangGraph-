# Agentic-AI-RAG-Chatbot-FAISS-Ollama-LangGraph-
<!DOCTYPE html>
<html lang="en">
<head>
    
   
</head>
<body>

<div class="container">

<h1>🤖 Agentic AI RAG Chatbot</h1>
<p>
This document explains the <strong>complete architecture</strong> and <strong>full code flow</strong>
for the Agentic AI RAG Chatbot built using <strong>LangGraph</strong>,
<strong>FAISS</strong>, and <strong>Ollama (free local AI)</strong>.
</p>

<hr>

<div class="section">
<h2>📌 Problem Statement</h2>
<p>
Build a RAG-based AI chatbot that answers questions strictly from a PDF knowledge base using:
</p>
<ul>
    <li>LangGraph for agentic workflow</li>
    <li>Vector database (FAISS)</li>
    <li>Local LLM and embeddings (Ollama)</li>
    <li>Chat API (FastAPI)</li>
    <li>Optional frontend (Streamlit)</li>
</ul>
</div>

<div class="section">
<h2>🧠 System Architecture</h2>
<pre>
PDF (Agentic AI eBook)
        ↓
Text Chunking
        ↓
Embeddings (Ollama - nomic-embed-text)
        ↓
FAISS Vector Store (Local)
        ↓
Retriever
        ↓
LangGraph Workflow
        ↓
LLM (Ollama - llama3)
        ↓
Final Answer + Context + Confidence
</pre>
</div>

<div class="section">
<h2>⚙️ Environment Setup (Anaconda)</h2>

<pre>
conda create -n agentic_rag python=3.10 -y
conda activate agentic_rag
</pre>

<pre>
pip install langchain langgraph langchain-community langchain-text-splitters \
faiss-cpu fastapi uvicorn pypdf streamlit requests python-dotenv
</pre>
</div>

<div class="section">
<h2>🧠 Ollama Setup (Free)</h2>

<pre>
ollama pull llama3
ollama pull nomic-embed-text
</pre>

<div class="note">
No API keys are required. All models run locally.
</div>
</div>

<div class="section">
<h2>📥 Ingestion Code (PDF → FAISS)</h2>

<pre>
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS

loader = PyPDFLoader("data/Ebook-Agentic-AI.pdf")
documents = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150
)
chunks = splitter.split_documents(documents)

embeddings = OllamaEmbeddings(model="nomic-embed-text")
vectorstore = FAISS.from_documents(chunks, embeddings)
vectorstore.save_local("vectorstore/faiss_index")

print("FAISS index created")
</pre>
</div>

<div class="section">
<h2>🧩 LangGraph State</h2>

<pre>
class RAGState(TypedDict):
    question: str
    context: list
    answer: str
    confidence: float
</pre>
</div>

<div class="section">
<h2>🔁 LangGraph Nodes</h2>

<pre>
def retrieve_node(state):
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    db = FAISS.load_local("vectorstore/faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = db.similarity_search(state["question"], k=3)
    return {"context": [d.page_content for d in docs]}

def generate_node(state):
    llm = Ollama(model="llama3", temperature=0)
    prompt = f"""
    Answer ONLY from the context below.
    If not found, say: Not found in knowledge base.
    Context: {state['context']}
    Question: {state['question']}
    """
    answer = llm.invoke(prompt)
    confidence = 0.9 if "Not found" not in answer else 0.3
    return {"answer": answer, "confidence": confidence}
</pre>
</div>

<div class="section">
<h2>🧠 LangGraph Workflow</h2>

<pre>
graph = StateGraph(RAGState)
graph.add_node("retrieve", retrieve_node)
graph.add_node("generate", generate_node)
graph.set_entry_point("retrieve")
graph.add_edge("retrieve", "generate")
app = graph.compile()
</pre>
</div>

<div class="section">
<h2>🌐 FastAPI Backend</h2>

<pre>
@app.post("/chat")
def chat(query: str):
    result = app.invoke({"question": query})
    return {
        "final_answer": result["answer"],
        "retrieved_context": result["context"],
        "confidence": result["confidence"]
    }
</pre>
</div>

<div class="section">
<h2>🎨 Streamlit Frontend</h2>

<pre>
query = st.text_input("Ask a question")
if st.button("Ask"):
    response = requests.post("http://127.0.0.1:8000/chat", params={"query": query})
    data = response.json()
    st.success(data["final_answer"])
    st.write("Confidence:", data["confidence"])
</pre>
</div>

<div class="section">
<h2>▶ How to Run</h2>

<pre>
python ingestion/ingest.py
uvicorn api.app:app --reload
streamlit run frontend/ui.py
</pre>
</div>

<div class="section">
<h2>✅ Key Features</h2>
<ul>
    <li>Grounded answers only</li>
    <li>Agentic workflow using LangGraph</li>
    <li>Local vector search with FAISS</li>
    <li>Free AI using Ollama</li>
    <li>API + UI support</li>
</ul>
</div>

<hr>

<p><strong>Author:</strong> Sparshi Jain</p>

</div>
</body>
</html>
