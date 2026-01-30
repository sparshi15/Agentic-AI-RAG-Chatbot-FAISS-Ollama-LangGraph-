from typing import TypedDict, List

class RAGState(TypedDict):
    question: str
    context: List[str]
    answer: str
    confidence: float
