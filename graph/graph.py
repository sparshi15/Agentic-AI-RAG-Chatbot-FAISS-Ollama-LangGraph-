from langgraph.graph import StateGraph
from graph.state import RAGState
from graph.nodes import retrieve_node, generate_node

def build_graph():
    graph = StateGraph(RAGState)

    graph.add_node("retrieve", retrieve_node)
    graph.add_node("generate", generate_node)

    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "generate")

    return graph.compile()
