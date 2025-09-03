# graph.py

import json
import operator
import os
import logging
from typing import Annotated, List, Literal, TypedDict

# --- Pydantic & DotEnv Imports ---
from pydantic import Field, create_model
from dotenv import load_dotenv

# --- LangChain & LangGraph Imports ---
from langchain_core.messages import (AIMessage, BaseMessage, HumanMessage, ToolMessage, SystemMessage)
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode


# --- External ToolRAGModel Package Imports ---
from txagent.toolrag import ToolRAGModel, SiliconFlowEmbeddingModel
from tooluniverse import ToolUniverse

# ==============================================================================
# --- 0. Environment and Logging Setup ---
# ==============================================================================
load_dotenv(override=True)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ==============================================================================
# --- 1. State Definition ---
# ==============================================================================
class AgentState(TypedDict):
    """Represents the state of the agent's conversation history."""
    messages: Annotated[List[BaseMessage], operator.add]


# ==============================================================================
# --- 2. Tools Abstraction & Initialization ---
# ==============================================================================
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
    return json.dumps(picked_tools)

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


# ==============================================================================
# --- 3. Graph Definition & Compilation ---
# ==============================================================================
def should_continue(state: AgentState) -> Literal["tools", "__end__"]:
    """Determines whether to continue the loop or end."""
    last_message = state["messages"][-1]
    return "tools" if isinstance(last_message, AIMessage) and last_message.tool_calls else END

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
        raise ValueError(f"Unsupported model specified: {model_name}")
        
    # Bind the RAG tool first so the model knows it can search for other tools
    model_with_rag_tool = model.bind_tools([Tool_RAG])

    def call_model(state: AgentState) -> dict[str, list[BaseMessage]]:
        """Invokes the LLM to get the next action."""
        logger.info("Agent is calling the model...")
        # Note: In a more advanced setup, you might bind different tools
        # based on the state. Here we consistently use the RAG tool.
        response = model_with_rag_tool.invoke(state["messages"])
        return {"messages": [response]}

    workflow = StateGraph(AgentState)
    workflow.add_node("agent", call_model)
    workflow.add_node("tools", tool_node)
    
    workflow.set_entry_point("agent")
    workflow.add_conditional_edges("agent", should_continue)
    workflow.add_edge("tools", "agent")
    
    return workflow.compile()