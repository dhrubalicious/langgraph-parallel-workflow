import os
import operator
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI

load_dotenv()

api_key = os.getenv("OPENROUTER_API_KEY")
if not api_key:
    raise RuntimeError("API key not found. Check your .env file.")


model = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key,
    model="nvidia/nemotron-3.5-super-vl:free",
    temperature=0.2,
)


# structured output
class EvaluationSchema(BaseModel):
    feedback: str = Field(
        description="Explain the evaluation, identify weaknesses, and provide actionable suggestions."
    )
    score: float = Field(description="score out of 10", ge=0, le=10)


structured_model = model.with_structured_output(EvaluationSchema)


# create the state
class AnswerAnalyzerState(TypedDict):
    answer: str
    accuracy: str
    depth: str
    communication: str
    individual_scores: Annotated[list[float], operator.add]
    avg_score: float


# functions
def accuracy_analysis(s):

    prompt = f"""
    Fact-check and critically evaluate the following answer for factual
    correctness, logical consistency, and false assumptions.

    Explicitly flag any verifiable inaccuracies, misleading claims,
    or outdated information, and provide the correct details.

    Score out of 10.

    Answer:
    {s["answer"]}
    """

    output = structured_model.invoke(prompt)

    return {"accuracy": output.feedback, "individual_scores": [output.score]}


def depth_analysis(s):

    prompt = f"""
    Critically evaluate the following answer for conceptual depth,
    understanding, and technical richness.

    Evaluate:
    1. Does the answer demonstrate a deep understanding of the topic?
    2. Does it explain underlying concepts rather than just definitions?
    3. Does it include relevant technical details and practical insights?
    4. Are important aspects missing?

    Provide constructive feedback and score out of 10.

    Answer:
    {s["answer"]}
    """

    output = structured_model.invoke(prompt)

    return {"depth": output.feedback, "individual_scores": [output.score]}


def communication_analysis(s):

    prompt = f"""
    Evaluate the following answer based on communication quality.

    Evaluate:
    1. Clarity and readability.
    2. Logical flow and structure.
    3. Conciseness (avoid unnecessary explanations).
    4. Whether the answer communicates ideas effectively.

    Identify weaknesses and suggest improvements.

    Provide feedback and score out of 10.

    Answer:
    {s["answer"]}
    """

    output = structured_model.invoke(prompt)

    return {"communication": output.feedback, "individual_scores": [output.score]}


def calculate_average(s):
    scores = s["individual_scores"]
    avg = sum(scores) / len(scores)

    return {"avg_score": round(avg, 2)}


# create the graph
graph = StateGraph(AnswerAnalyzerState)

# nodes and edges
graph.add_node("accuracy_analysis", accuracy_analysis)
graph.add_node("depth_analysis", depth_analysis)
graph.add_node("communication_analysis", communication_analysis)
graph.add_node("final_score", calculate_average)

# edges
graph.add_edge(START, "accuracy_analysis")
graph.add_edge(START, "depth_analysis")
graph.add_edge(START, "communication_analysis")

graph.add_edge("accuracy_analysis", "final_score")
graph.add_edge("depth_analysis", "final_score")
graph.add_edge("communication_analysis", "final_score")
graph.add_edge("final_score", END)

answer = """RAG stands for Retrieval-Augmented Generation. It is a technique used to improve the responses of Large Language Models by connecting them to external knowledge sources.

In RAG, documents are first divided into smaller chunks, which are converted into vector embeddings and stored in a vector database. When a user asks a question, the system converts the query into an embedding and searches the database for similar vectors.

The retrieved documents are then passed to the LLM along with the original question. The LLM uses this additional context to generate a relevant answer.

RAG helps reduce hallucinations and allows LLMs to access updated information without retraining the entire model.
"""

workflow = graph.compile()
initial_state = {"answer": answer}
result = workflow.invoke(initial_state)

print(result)
