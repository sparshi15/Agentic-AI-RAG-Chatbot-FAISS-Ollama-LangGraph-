from fastapi import FastAPI
from graph.graph import build_graph

app = FastAPI()
graph = build_graph()

@app.post("/chat")
def chat(query: str):
    result = graph.invoke({"question": query})

    return {
        "final_answer": result["answer"],
        "retrieved_context": result["context"],
        "confidence": result["confidence"]
    }
