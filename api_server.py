import os
import uvicorn
import logging
import argparse
import requests # <-- Added for making HTTP requests
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from typing import List, Dict, Any, Optional, Literal, Union, TypedDict

# --- Pydantic Models for API Structure ---
from pydantic import BaseModel, ConfigDict, model_validator

# --- LangChain & LangGraph Imports ---
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel as V1BaseModel, Field as V1Field
from langgraph.graph import StateGraph, END

# ==============================================================================
# --- 0. Environment and Logging Setup ---
# ==============================================================================
load_dotenv(override=True)
os.environ["MKL_THREADING_LAYER"] = "GNU"
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Check for OpenAI API Key
if not os.getenv("OPENAI_API_KEY"):
    logger.error("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
    exit(1)

# ==============================================================================
# --- 1. Standardized API Data Structures (As Provided) ---
# ==============================================================================
class ContentBlock(BaseModel):
    model_config = ConfigDict(extra="allow")
    type: Literal["text", "image", "file", "audio"]
    text: Optional[str] = None
    source_type: Optional[Literal["base64", "url"]] = None
    data: Optional[str] = None
    mime_type: Optional[str] = None
    url: Optional[str] = None
    @model_validator(mode='after')
    def check_content_consistency(self) -> 'ContentBlock':
        if self.type == 'text':
            if self.text is None: raise ValueError("For a 'text' block, the 'text' field is required.")
            if self.source_type or self.data or self.mime_type or self.url: raise ValueError("For a 'text' block, only the 'text' field should be provided.")
        else:
            if self.text is not None: raise ValueError(f"For a '{self.type}' block, the 'text' field should not be present.")
            if self.source_type == 'base64':
                if self.data is None or self.mime_type is None: raise ValueError("For a 'base64' source, 'data' and 'mime_type' are required.")
            elif self.source_type == 'url':
                if self.url is None: raise ValueError("For a 'url' source, the 'url' field is required.")
            else:
                raise ValueError(f"For a '{self.type}' block, 'source_type' must be 'base64' or 'url'.")
        return self

class Message(BaseModel):
    role: Literal["user", "assistant", "system", "tool"]
    content: Union[str, List[ContentBlock]]

class AgentRequest(BaseModel):
    request_id: str
    messages: List[Message]
    metadata: Optional[Dict[str, Any]] = None

class AgentResponse(BaseModel):
    request_id: str
    answer: str
    complete_messages: Optional[List[Message]] = None

# ==============================================================================
# --- 2. Medical Agent Workflow Class ---
# ==============================================================================
class MedicalAgent:
    """
    Encapsulates the LangGraph workflow for the medical Q&A system.
    """
    # --- Pydantic Models for Structured LLM Output ---
    class Entities(V1BaseModel):
        """Represents the medical entities extracted from a question."""
        entities: List[str] = V1Field(
            description="A list of key medical terms, product names, or concepts found in the user's question."
        )

    # --- Graph State Definition ---
    class GraphState(TypedDict):
        original_question: str
        entities: List[str]
        retrieved_documents: List[str]
        reasoning_log: List[str]
        summary: str
        final_answer: str
        iteration_count: int

    def __init__(self, mcp_url: str = "http://127.0.0.1:9000/mcp"):
        logger.info("Initializing MedicalAgent workflow...")
        self.mcp_url = mcp_url
        workflow = StateGraph(self.GraphState)

        # Add nodes
        workflow.add_node("entity_recognition_agent", self._entity_recognition_agent)
        workflow.add_node("search_agent", self._search_agent)
        workflow.add_node("reasoning_agent", self._reasoning_agent)
        workflow.add_node("summarizer_agent", self._summarizer_agent)
        workflow.add_node("answer_agent", self._answer_agent)

        # Set entry point and edges
        workflow.set_entry_point("entity_recognition_agent")
        workflow.add_edge("entity_recognition_agent", "search_agent")
        workflow.add_edge("search_agent", "reasoning_agent")
        workflow.add_conditional_edges(
            "reasoning_agent",
            self._should_continue,
            {"continue": "search_agent", "end": "summarizer_agent"}
        )
        workflow.add_edge("summarizer_agent", "answer_agent")
        workflow.add_edge("answer_agent", END)

        # Compile the graph
        self.app = workflow.compile()
        logger.info("MedicalAgent workflow compiled successfully.")

    # --- Agent Node Methods ---
    def _entity_recognition_agent(self, state: GraphState):
        logger.info("Node: Entity Recognition")
        prompt = ChatPromptTemplate.from_template(
            "You are a medical expert. Identify all key medical entities from the following question.\n\n"
            "Question: {question}"
        )
        llm = ChatOpenAI(model="gpt-4o", temperature=0)
        structured_llm = llm.with_structured_output(self.Entities)
        chain = prompt | structured_llm
        response = chain.invoke({"question": state['original_question']})
        logger.info(f"Entities Found: {response.entities}")
        return {
            "entities": response.entities,
            "iteration_count": 0,
            "reasoning_log": []
        }

    def _search_agent(self, state: GraphState):
        """
        *** UPDATED METHOD ***
        Calls the ToolUniverse MCP to perform a real search.
        """
        logger.info(f"Node: Search (Iteration {state['iteration_count'] + 1})")
        current_iteration = state['iteration_count'] + 1
        
        # Create a search query from entities
        search_query = " ".join(state['entities'])
        logger.info(f"Constructed Search Query: {search_query}")
        
        retrieved_docs = []
        try:
            # Call the ToolUniverse MCP API
            # NOTE: The payload format `{"query": ...}` is an assumption.
            # Adjust if your tool expects a different format.
            payload = {"query": search_query}
            response = requests.post(self.mcp_url, json=payload, timeout=15)
            response.raise_for_status() # Raises an HTTPError for bad responses (4xx or 5xx)
            
            # NOTE: The response parsing is an assumption.
            # We assume the response is a JSON object with a key 'results' containing a list of strings.
            # Adjust if your tool returns a different structure.
            results = response.json().get("results", [])
            if isinstance(results, list):
                retrieved_docs = [str(doc) for doc in results]
            else:
                logger.warning(f"MCP response 'results' is not a list: {results}")

            logger.info(f"Retrieved {len(retrieved_docs)} documents from MCP.")

        except requests.exceptions.RequestException as e:
            logger.error(f"Error calling ToolUniverse MCP: {e}")
            # Return empty list on error to allow the workflow to continue gracefully
            retrieved_docs = ["Search failed due to a network error."]
            
        except Exception as e:
            logger.error(f"An unexpected error occurred during search: {e}")
            retrieved_docs = ["Search failed due to an unexpected error."]

        return {
            "retrieved_documents": retrieved_docs,
            "iteration_count": current_iteration
        }

    def _reasoning_agent(self, state: GraphState):
        logger.info("Node: Reasoning")
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a medical researcher. Determine if the provided information is sufficient to answer the user's question. Respond with 'yes' or 'no'."),
            ("human", "User Question: {question}\n\nRetrieved Information:\n{documents}")
        ])
        llm = ChatOpenAI(model="gpt-4o", temperature=0)
        chain = prompt | llm
        response = chain.invoke({"question": state['original_question'], "documents": "\n\n".join(state['retrieved_documents'])})
        decision = response.content.strip().lower()
        logger.info(f"Reasoning decision: {decision}")
        log_entry = f"Iteration {state['iteration_count']}: Decision is '{decision}'."
        return {"reasoning_log": state['reasoning_log'] + [log_entry]}

    def _summarizer_agent(self, state: GraphState):
        logger.info("Node: Summarizer")
        prompt = ChatPromptTemplate.from_template(
            "Create a concise summary from the retrieved documents that directly addresses the user's question.\n\n"
            "User Question: {question}\n\nRetrieved Documents:\n{documents}"
        )
        llm = ChatOpenAI(model="gpt-4o", temperature=0)
        chain = prompt | llm
        summary = chain.invoke({"question": state['original_question'], "documents": "\n\n".join(state['retrieved_documents'])}).content
        return {"summary": summary}

    def _answer_agent(self, state: GraphState):
        logger.info("Node: Answer Generation")
        prompt = ChatPromptTemplate.from_template(
            "Based on the summary, answer the user's question. For multiple-choice questions, select the correct option (e.g., 'C').\n\n"
            "Summary:\n{summary}\n\nUser Question: {question}\nOptions: A: 5, B: 10, C: 15, D: 30"
        )
        llm = ChatOpenAI(model="gpt-4o", temperature=0)
        chain = prompt | llm
        response = chain.invoke({"summary": state['summary'], "question": state['original_question']}).content
        return {"final_answer": response}

    # --- Conditional Edge Method ---
    def _should_continue(self, state: GraphState):
        logger.info("Conditional Edge: Should Continue?")
        last_decision = state['reasoning_log'][-1]
        if "no" in last_decision.lower() and state['iteration_count'] < 3:
            logger.info("Decision: Continue.")
            return "continue"
        else:
            logger.info("Decision: End loop.")
            return "end"

    # --- Public Method to Run the Workflow ---
    def run(self, question: str) -> str:
        logger.info(f"Running workflow for question: '{question}'")
        initial_state = {"original_question": question}
        final_state = self.app.invoke(initial_state)
        logger.info("Workflow complete.")
        return final_state.get('final_answer', "Error: Could not generate an answer.")

# ==============================================================================
# --- 3. FastAPI Server Setup ---
# ==============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Server starting up...")
    # Pass the MCP URL from CLI args or a default
    mcp_url = getattr(cli_args, 'mcp_url', 'http://127.0.0.1:9000/mcp')
    app.state.agent = MedicalAgent(mcp_url=mcp_url)
    yield
    logger.info("Server shutting down...")

app = FastAPI(lifespan=lifespan)

@app.post("/invoke", response_model=AgentResponse)
async def invoke(request: AgentRequest):
    logger.info(f"Received request with ID: {request.request_id}")
    if not request.messages:
        raise HTTPException(status_code=400, detail="The 'messages' list cannot be empty.")

    last_message = request.messages[-1]
    if last_message.role != 'user' or not isinstance(last_message.content, str):
        raise HTTPException(status_code=400, detail="The last message must be from the 'user' and have string content.")
    
    question = last_message.content
    
    try:
        answer_text = app.state.agent.run(question)
        return AgentResponse(request_id=request.request_id, answer=answer_text)
    except Exception as e:
        logger.error(f"An error occurred while processing request {request.request_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An internal error occurred in the agent.")

@app.get("/health")
def health_check():
    return {"status": "ok"}

# ==============================================================================
# --- 4. Command-Line Interface and Server Execution ---
# ==============================================================================
# Global variable to hold parsed args for lifespan function
cli_args = None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Medical Agent API Server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="The host to run the server on.")
    parser.add_argument("--port", type=int, default=8000, help="The port to run the server on.")
    parser.add_argument("--mcp_url", type=str, default="http://127.0.0.1:9000/mcp", help="The URL for the ToolUniverse MCP service.")
    
    # Parse arguments and store them globally
    cli_args = parser.parse_args()

    logger.info(f"Starting server at http://{cli_args.host}:{cli_args.port}")
    logger.info(f"Using ToolUniverse MCP at: {cli_args.mcp_url}")
    uvicorn.run(app, host=cli_args.host, port=cli_args.port)
