from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS


# Load PDF
loader = PyPDFLoader("data/Ebook-Agentic-AI.pdf")
documents = loader.load()

# Split text
splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150
)
chunks = splitter.split_documents(documents)

# Ollama embeddings (FREE, local)
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# Create FAISS vector store
vectorstore = FAISS.from_documents(chunks, embeddings)

# Save FAISS index
vectorstore.save_local("vectorstore/faiss_index")

print("✅ FAISS index created successfully")

