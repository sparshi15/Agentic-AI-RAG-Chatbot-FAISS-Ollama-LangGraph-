from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama

def retrieve_node(state):
    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    db = FAISS.load_local(
        "vectorstore/faiss_index",
        embeddings,
        allow_dangerous_deserialization=True
    )

    docs = db.similarity_search(state["question"], k=3)

    return {"context": [d.page_content for d in docs]}


def generate_node(state):
    llm = Ollama(model="llama3", temperature=0)

    prompt = f"""
Answer ONLY from the context below.
If the answer is not present, say:
Not found in knowledge base.

Context:
{state['context']}

Question:
{state['question']}
"""

    answer = llm.invoke(prompt)
    confidence = 0.9 if "Not found" not in answer else 0.3

    return {
        "answer": answer,
        "confidence": confidence
    }
