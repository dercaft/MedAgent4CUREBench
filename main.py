import os
from typing import List, TypedDict, Annotated
import operator

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langgraph.graph import StateGraph, END

# --- Environment Setup ---
# Set your OpenAI API key here. It's recommended to use environment variables.
# os.environ["OPENAI_API_KEY"] = "YOUR_API_KEY"

# --- 1. Define the State for the Graph ---
# This TypedDict will hold the state of our workflow as it progresses through the graph.
class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        original_question: The initial question asked by the user.
        entities: Medical entities extracted from the question.
        retrieved_documents: Information retrieved from our search tool.
        reasoning_log: A log of the reasoning process and decisions.
        summary: A summary of all gathered information.
        final_answer: The final answer to the question.
        iteration_count: The number of search-reasoning loops.
    """
    original_question: str
    entities: List[str]
    retrieved_documents: List[str]
    reasoning_log: List[str]
    summary: str
    final_answer: str
    iteration_count: int

# --- (Optional) Pydantic Models for Structured Output ---
# We can use Pydantic to force the LLM to return structured data, which is more reliable.
class Entities(BaseModel):
    """Represents the medical entities extracted from a question."""
    entities: List[str] = Field(
        description="A list of key medical terms, product names, or concepts found in the user's question."
    )

# --- 2. Define the Agent Nodes ---
# Each node in our graph is a function that takes the current state and returns an updated state.

def entity_recognition_agent(state: GraphState):
    """
    Identifies key medical entities in the user's question and initializes the state.
    """
    print("--- 1. EXECUTING ENTITY RECOGNITION AGENT ---")
    
    prompt = ChatPromptTemplate.from_template(
        "You are a medical expert. Identify all key medical entities from the following question. "
        "Do not infer or add information not present in the text.\n\n"
        "Question: {question}"
    )
    
    # Initialize the LLM
    # Using a model that supports structured output (function calling) is best.
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    structured_llm = llm.with_structured_output(Entities)
    
    chain = prompt | structured_llm
    
    question = state['original_question']
    response = chain.invoke({"question": question})
    
    print(f"Entities Found: {response.entities}")
    
    # *** FIX: Return all initialized state variables ***
    # The node must return a dictionary with all the keys it intends to update in the state.
    return {
        "entities": response.entities,
        "iteration_count": 0, # Initialize count
        "reasoning_log": []   # Initialize log
    }

def search_agent(state: GraphState):
    """
    Simulates a search query based on the identified entities.
    In a real system, this would call a search API (e.g., PubMed, Tavily).
    """
    print("--- 2. EXECUTING SEARCH AGENT ---")
    
    # Now this line will work correctly
    current_iteration = state['iteration_count'] + 1
    
    entities = state['entities']
    
    # --- MOCK SEARCH ---
    # For this MVP, we simulate the search result.
    # A real implementation would use the entities to form a query.
    print(f"Searching with entities: {entities}")
    mock_search_result = (
        "According to the official guidelines for the '70%' brand sunscreen, "
        "it is recommended to use a product with a Sun Protection Factor (SPF) of at least 15. "
        "This value is considered the minimum for effective protection against UV-induced skin damage, "
        "which helps reduce the long-term risk of skin cancer and premature skin aging."
    )
    
    print(f"Documents Retrieved: [Simulated Document]")
    
    return {
        "retrieved_documents": [mock_search_result],
        "iteration_count": current_iteration
    }

def reasoning_agent(state: GraphState):
    """
    Analyzes the retrieved documents to see if they are sufficient to answer the question.
    """
    print("--- 3. EXECUTING REASONING AGENT ---")
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", 
         "You are a medical researcher. Your task is to determine if the provided information "
         "is sufficient to definitively answer the user's question. "
         "Respond with 'yes' if the information is sufficient, or 'no' if more information is needed."),
        ("human", 
         "User Question: {question}\n\n"
         "Retrieved Information:\n{documents}")
    ])
    
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    chain = prompt | llm
    
    question = state['original_question']
    documents = "\n\n".join(state['retrieved_documents'])
    
    response = chain.invoke({"question": question, "documents": documents})
    decision = response.content.strip().lower()
    
    print(f"Is the information sufficient? {decision}")
    
    log_entry = f"Iteration {state['iteration_count']}: Decision is '{decision}'."
    
    return {"reasoning_log": state['reasoning_log'] + [log_entry]}


def summarizer_agent(state: GraphState):
    """
    Summarizes the gathered information into a concise paragraph.
    """
    print("--- 4. EXECUTING SUMMARIZER AGENT ---")
    
    prompt = ChatPromptTemplate.from_template(
        "You are a medical report writer. Based on the following retrieved documents, "
        "create a concise summary that directly addresses the user's original question.\n\n"
        "User Question: {question}\n\n"
        "Retrieved Documents:\n{documents}"
    )
    
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    chain = prompt | llm
    
    question = state['original_question']
    documents = "\n\n".join(state['retrieved_documents'])
    
    summary = chain.invoke({"question": question, "documents": documents}).content
    
    print(f"Generated Summary: {summary}")
    
    return {"summary": summary}

def answer_agent(state: GraphState):
    """
    Generates the final answer based on the summary.
    For multiple-choice questions, it selects the best option.
    """
    print("--- 5. EXECUTING ANSWER AGENT ---")
    
    prompt = ChatPromptTemplate.from_template(
        "You are a helpful medical AI assistant. Based on the provided summary, "
        "answer the user's original question. For multiple-choice questions, "
        "select the correct option (e.g., 'C').\n\n"
        "Summary:\n{summary}\n\n"
        "User Question: {question}\n"
        "Options: {options}"
    )
    
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    chain = prompt | llm
    
    # In a real app, options would be part of the input. We'll hardcode them for this example.
    options_str = "A: 5, B: 10, C: 15, D: 30"
    
    response = chain.invoke({
        "summary": state['summary'],
        "question": state['original_question'],
        "options": options_str
    }).content
    
    print(f"Final Answer: {response}")
    
    return {"final_answer": response}

# --- 3. Define Conditional Edges ---
# This function decides the next step after the reasoning agent.
def should_continue(state: GraphState):
    """
    Determines whether to continue the search-reason loop or to proceed to summarization.
    """
    print("--- DECISION POINT ---")
    
    # Check the last entry in the reasoning log
    last_decision = state['reasoning_log'][-1]
    
    # Simple logic: if the LLM said 'no', and we haven't looped too many times, continue.
    if "no" in last_decision.lower() and state['iteration_count'] < 3:
        print("Decision: Information is insufficient. Looping back to search.")
        return "continue"
    else:
        print("Decision: Information is sufficient. Proceeding to summarize.")
        return "end"

# --- 4. Build the Graph ---
# We now wire together the nodes and edges to create the workflow.

# Initialize a new graph
workflow = StateGraph(GraphState)

# Add the nodes
workflow.add_node("entity_recognition_agent", entity_recognition_agent)
workflow.add_node("search_agent", search_agent)
workflow.add_node("reasoning_agent", reasoning_agent)
workflow.add_node("summarizer_agent", summarizer_agent)
workflow.add_node("answer_agent", answer_agent)

# Set the entry point of the graph
workflow.set_entry_point("entity_recognition_agent")

# Add the edges
workflow.add_edge("entity_recognition_agent", "search_agent")
workflow.add_edge("search_agent", "reasoning_agent")

# Add the conditional edge from the reasoning agent
workflow.add_conditional_edges(
    "reasoning_agent",
    should_continue,
    {
        "continue": "search_agent", # If 'continue', loop back to search
        "end": "summarizer_agent"     # If 'end', proceed to summarize
    }
)

workflow.add_edge("summarizer_agent", "answer_agent")
workflow.add_edge("answer_agent", END) # The final step

# Compile the graph into a runnable object
app = workflow.compile()

# --- 5. Run the System ---
if __name__ == "__main__":
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OpenAI API key not found.")
        print("Please set the OPENAI_API_KEY environment variable.")
    else:
        # The input data from your example
        data_sample = {
            "question": "What is the minimum SPF value recommended for sunscreens to reduce the risk of skin cancer and early aging, as per the guidelines for the sunscreen branded as '70%'?",
        }

        # The initial state for the graph
        initial_state = {"original_question": data_sample["question"]}
        
        print("--- STARTING MEDICAL AGENT WORKFLOW ---")
        # Invoke the graph and stream the results
        final_state = app.invoke(initial_state)
        print("\n--- WORKFLOW COMPLETE ---")
        
        print("\nFinal Answer from the graph:")
        print(final_state['final_answer'])
