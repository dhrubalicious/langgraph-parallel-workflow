# Parallel Answer Analyzer using LangGraph

A simple AI-powered answer evaluation system built using **LangGraph**, **LangChain**, and **OpenRouter**.

This project demonstrates how parallel workflows work in LangGraph by evaluating a given answer across three independent dimensions using LLM calls.

## How It Works

The user provides an answer, which is then evaluated by three independent LLM-powered nodes executing in parallel.

### Evaluation Dimensions

**1. Accuracy Analysis**

* Evaluates factual correctness and logical consistency.
* Identifies misleading claims, inaccuracies, and false assumptions.
* Assigns a score out of 10.

**2. Depth Analysis**

* Evaluates conceptual understanding and technical depth.
* Checks whether the answer explains underlying concepts.
* Identifies missing technical details and practical insights.
* Assigns a score out of 10.

**3. Communication Analysis**

* Evaluates clarity, readability, and logical structure.
* Checks conciseness and effectiveness of communication.
* Provides suggestions for improvement.
* Assigns a score out of 10.

### Final Score Aggregation

Once all three parallel branches finish, a final node collects their individual scores and calculates the average score.

## Workflow Architecture

```text
                  START
                    |
          +---------+---------+
          |         |         |
          v         v         v
       Accuracy   Depth   Communication
       Analysis  Analysis   Analysis
          |         |         |
          +---------+---------+
                    |
                    v
             Final Score
              Aggregation
                    |
                    v
                   END
```

## Technologies Used

* Python
* LangGraph
* LangChain
* Pydantic
* OpenRouter API

## Key Concepts Demonstrated

* Parallel execution in LangGraph
* State management using TypedDict
* Reducers using `operator.add`
* Structured LLM outputs using Pydantic
* Fan-out and fan-in workflow architecture
* Aggregation of results from parallel branches

## Note

This is a small learning project created to understand and experiment with parallel workflows in LangGraph.

