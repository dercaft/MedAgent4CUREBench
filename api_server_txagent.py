import json
import operator
import os
import uvicorn
import logging
from typing import Annotated, List, Dict, Any, Optional, Literal, Union, Tuple, TypedDict
from contextlib import asynccontextmanager

# --- FastAPI and Pydantic Imports ---
from fastapi import FastAPI, HTTPException, Request, Depends # <-- Add Depends
from pydantic import BaseModel, ConfigDict, model_validator, Field, create_model

# --- LangChain & LangGraph Imports ---
from langchain_core.messages import (AIMessage, BaseMessage, HumanMessage, ToolMessage, SystemMessage)
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode

# --- External ToolRAGModel Package Imports ---
from dotenv import load_dotenv
from txagent.toolrag import ToolRAGModel, SiliconFlowEmbeddingModel
from tooluniverse import ToolUniverse
from fastapi import Request

async def log_request_body(request: Request):
    try:
        body = await request.json()
        logger.info(f"Incoming request body: {body}")
    except Exception:
        logger.warning("Could not parse request body.")
# ==============================================================================
# --- 0. Environment and Logging Setup ---
# ==============================================================================
load_dotenv(override=True)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ==============================================================================
# --- 1. Standardized API Data Structures ---
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
        # This agent only supports text, so we don't implement the multimodal checks.
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
# --- 2. LangGraph Agent Core Logic (from agent_main_v2.py) ---
# ==============================================================================

# --- Part 2.1: State Definition ---
class AgentState(TypedDict):
    """Represents the state of the agent's conversation history."""
    messages: Annotated[List[BaseMessage], operator.add]

# --- Part 2.2: Tools Abstraction & Initialization ---
# This section contains global instances that will be initialized once.
logger.info("Initializing ToolUniverse and ToolRAGModel...")
current_dir = os.path.dirname(os.path.abspath(__file__))
default_tool_files = {
    'opentarget': os.path.join(current_dir, 'txagent', 'data', 'opentarget_tools.json'),
    'fda_drug_label': os.path.join(current_dir, 'txagent', 'data', 'fda_drug_labeling_tools.json'),
    'monarch': os.path.join(current_dir, 'txagent', 'data', 'monarch_tools.json')
}
tool_universe = ToolUniverse(tool_files=default_tool_files, keep_default_tools=False)
tool_universe.load_tools()

RAG_MODEL_NAME = "Qwen/Qwen3-Embedding-8B"
embedding_model = SiliconFlowEmbeddingModel(RAG_MODEL_NAME)
tool_rag_model = ToolRAGModel(embedding_model)
tool_rag_model.load_tool_desc_embedding(tool_universe)
logger.info("Tool models initialized successfully.")

@tool
def Tool_RAG(query: str, rag_num: int = 5) -> str:
    """
    Retrieves a list of relevant tools for a given query. Use this when you are 
    unsure which specific tool to use or need to discover tools for a task.
    """
    logger.info(f"🔎 RAG searching for tools with query: '{query}'")
    top_tool_names = tool_rag_model.rag_infer(query, top_k=rag_num)
    picked_tools = tool_universe.get_tool_by_name(top_tool_names)
    if not picked_tools:
        return "No relevant tools were found for the query."
    return json.dumps(picked_tools, indent=2)

def load_all_tools() -> List[callable]:
    """
    Loads the RAG tool and dynamically creates all other tools from the ToolUniverse.
    """
    all_tools: List[callable] = [Tool_RAG]
    type_map = {"string": str, "integer": int, "number": float, "boolean": bool}

    for tool_schema in tool_universe.all_tools:
        if tool_schema['name'] == 'Tool_RAG':
            continue

        params = tool_schema.get('parameter', {})
        properties = params.get('properties', {})
        required_params = params.get('required', [])
        
        fields = {}
        for param_name, param_details in properties.items():
            param_type = type_map.get(param_details.get("type"), str)
            description = param_details.get("description", "")
            if param_name in required_params:
                fields[param_name] = (param_type, Field(..., description=description))
            else:
                fields[param_name] = (param_type, Field(default=None, description=description))

        args_schema = create_model(tool_schema['name'] + 'Input', **fields)

        def create_tool_from_schema(schema: dict) -> callable:
            def _tool_func(**kwargs):
                return tool_universe.run_one_function({
                    "name": schema['name'],
                    "arguments": kwargs
                })
            _tool_func.__name__ = schema['name']
            _tool_func.__doc__ = schema['description']
            return tool(_tool_func, args_schema=args_schema)

        all_tools.append(create_tool_from_schema(tool_schema))
    
    logger.info(f"Loaded a total of {len(all_tools)} tools.")
    return all_tools

# --- Part 2.3: Graph Definition ---
def should_continue(state: AgentState) -> Literal["tools", "__end__"]:
    """Determines whether to continue the loop or end."""
    return "tools" if isinstance(state["messages"][-1], AIMessage) and state["messages"][-1].tool_calls else END

def build_graph(model_name: str) -> CompiledStateGraph:
    """Builds and compiles the LangGraph agent."""
    if "OPENAI_API_KEY" not in os.environ and "GOOGLE_API_KEY" not in os.environ:
        raise ValueError("OPENAI_API_KEY or GOOGLE_API_KEY not found in environment.")

    all_tools = load_all_tools()
    tool_node = ToolNode(all_tools)

    if "gpt" in model_name.lower():
        model = ChatOpenAI(model=model_name, temperature=0.1, max_tokens=2048)
    elif "gemini" in model_name.lower():
        api_endpoint = os.getenv("GOOGLE_BASE_URL")
        model = ChatGoogleGenerativeAI(model=model_name, client_options={"api_endpoint": api_endpoint})
    else:
        raise ValueError("Unsupported model specified in build_graph.")
        
    model_with_rag_tool = model.bind_tools([Tool_RAG])

    def call_model(state: AgentState) -> dict[str, list[BaseMessage]]:
        """Invokes the LLM to get the next action."""
        logger.info("Agent is calling the model...")
        response = model_with_rag_tool.invoke(state["messages"])
        return {"messages": [response]}

    workflow = StateGraph(AgentState)
    workflow.add_node("agent", call_model)
    workflow.add_node("tools", tool_node)
    
    workflow.set_entry_point("agent")
    workflow.add_conditional_edges("agent", should_continue)
    workflow.add_edge("tools", "agent")
    
    return workflow.compile()

# ==============================================================================
# --- 3. FastAPI Application Setup ---
# ==============================================================================

# --- Part 3.1: Lifespan Manager for Model & Graph Loading ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load the graph and store it in the app state
    logger.info("FastAPI app startup: Building LangGraph agent...")
    model_name = os.getenv("MODEL_NAME", "gemini-1.5-flash") # Default to gemini
    app.state.graph = build_graph(model_name)
    logger.info(f"LangGraph agent with model '{model_name}' built successfully.")
    yield
    # Shutdown: Clean up resources if any
    logger.info("FastAPI app shutdown.")

app = FastAPI(
    title="LangGraph RAG Agent API",
    description="An API server for a LangGraph agent with dynamic tool retrieval.",
    version="1.0.0",
    lifespan=lifespan
)

# --- Part 3.2: Helper Functions for Message Conversion ---
def convert_api_messages_to_langchain(messages: List[Message]) -> List[BaseMessage]:
    """Converts API message format to LangChain's BaseMessage format."""
    lc_messages = []
    for msg in messages:
        lc_content = None
        if isinstance(msg.content, str):
            lc_content = msg.content
        elif isinstance(msg.content, list): 
            # This should be a list of ContentBlock objects, convert to List[Dict]
            lc_content = []
            for block in msg.content:
                lc_content.append(block.model_dump())
        else:
            raise HTTPException(status_code=400, detail="Invalid message content type.")
        
        if msg.role == "user":
            lc_messages.append(HumanMessage(content=lc_content))
        elif msg.role == "assistant":
            lc_messages.append(AIMessage(content=lc_content))
        elif msg.role == "system":
            lc_messages.append(SystemMessage(content=lc_content))
        # Note: Converting 'tool' messages would require more info like 'tool_call_id'.
        # This implementation assumes a simple user/assistant history as input.
    return lc_messages

def convert_langchain_messages_to_api(messages: List[BaseMessage]) -> List[Message]:
    """Converts LangChain's BaseMessage format back to API message format."""
    api_messages = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            role = "user"
        elif isinstance(msg, AIMessage):
            role = "assistant"
        elif isinstance(msg, SystemMessage):
            role = "system"
        elif isinstance(msg, ToolMessage):
            role = "tool"
        else:
            continue # Skip unknown message types
        
        api_messages.append(Message(role=role, content=str(msg.content)))
    return api_messages


# --- Part 3.3: API Endpoint ---
# In api_server.py

@app.post("/invoke", response_model=AgentResponse, dependencies=[Depends(log_request_body)]) # <-- Add this
async def invoke_agent(agent_request: AgentRequest, http_request: Request):
    """
    Receives a request, invokes the LangGraph agent, and returns the final response.
    """
    logger.info(f"Received request_id: {agent_request.request_id}")
    
    # 1. Retrieve the compiled graph from the application state
    graph = http_request.app.state.graph
    if not graph:
        raise HTTPException(status_code=500, detail="Graph is not initialized.")

    # 2. Convert incoming messages to LangChain format
    try:
        input_messages = convert_api_messages_to_langchain(agent_request.messages)
    except HTTPException as e:
        raise e
    
    # --- ⬇️ START: ADDED LOGIC ⬇️ ---

    # 3. Check for and apply the system prompt from metadata
    system_prompt = None
    if agent_request.metadata and "system_prompt" in agent_request.metadata:
        system_prompt = agent_request.metadata["system_prompt"]

    if system_prompt and isinstance(system_prompt, str):
        logger.info("Applying system prompt from request metadata.")
        system_message = SystemMessage(content=system_prompt)
        # Prepend the system message to the beginning of the conversation
        input_messages.insert(0, system_message)

    # --- ⬆️ END: ADDED LOGIC ⬆️ ---
    
    # 4. Prepare the input for the graph
    inputs = {"messages": input_messages}
    
    # 5. Invoke the graph and get the final state
    try:
        final_state = graph.invoke(inputs)
        
        # 6. Extract the final answer and complete message history
        final_messages_lc = final_state.get("messages", [])
        if not final_messages_lc or not isinstance(final_messages_lc[-1], AIMessage):
             raise HTTPException(status_code=500, detail="Agent did not produce a final answer.")
             
        final_answer = final_messages_lc[-1].content
        complete_messages_api = convert_langchain_messages_to_api(final_messages_lc)
        
        logger.info(f"Successfully processed request_id: {agent_request.request_id}")
        
        # 7. Return the structured response
        return AgentResponse(
            request_id=agent_request.request_id,
            answer=final_answer,
            complete_messages=complete_messages_api
        )
    except Exception as e:
        logger.error(f"Error invoking graph for request_id {agent_request.request_id}: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred during agent execution: {e}")

@app.get("/", tags=["Health Check"])
def health_check():
    """
    Provides a simple health check to confirm the API is running and see model info.
    """
    # Retrieve the model name that was used at startup
    model_name = os.getenv("MODEL_NAME", "gemini-1.5-flash")
    
    return {
        "status": "ok",
        "message": "LangGraph RAG Agent API is running.",
        "details": {
            "active_llm": model_name,
            "rag_embedding_model": RAG_MODEL_NAME
        }
    }
# ==============================================================================
# --- 4. Main Execution Block ---
# ==============================================================================
if __name__ == "__main__":
    # To run this server:
    # 1. Make sure you have an .env file with GOOGLE_API_KEY or OPENAI_API_KEY.
    # 2. Run in your terminal: uvicorn api_server:app --host 0.0.0.0 --port 8000 --reload
    uvicorn.run(app, host="127.0.0.1", port=8128)