# agent_main_v2.py (Refactored)
"""
A self-contained script for a LangGraph agent that uses dynamic tool retrieval.
This version is refactored based on modern LangGraph documentation, using the
@tool decorator and a simplified graph structure.
"""
import json
import operator
import os
from typing import Annotated, List, Literal, Tuple, TypedDict
from pydantic import BaseModel, Field, create_model

from dotenv import load_dotenv
from langchain_core.messages import (AIMessage, BaseMessage, HumanMessage,
                                     ToolMessage)
# The @tool decorator is the modern way to define tools
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI


from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode

# --- Import the external ToolRAGModel package ---
from txagent.toolrag import ToolRAGModel, SiliconFlowEmbeddingModel
from tooluniverse import ToolUniverse

# --- Part 1: State Definition (Unchanged) ---

class AgentState(TypedDict):
    """Represents the state of the agent's conversation history."""
    messages: Annotated[List[BaseMessage], operator.add]

# --- Part 2: Tools Abstraction (Refactored) ---

# --- Global Instantiation and Initialization ---
load_dotenv(override=True)
# Set tooluniverse to only include the tools we want to use
current_dir = os.path.dirname(os.path.abspath(__file__))
default_tool_files = {
    'opentarget': os.path.join(current_dir, 'txagent', 'data', 'opentarget_tools.json'),
    'fda_drug_label': os.path.join(current_dir, 'txagent', 'data', 'fda_drug_labeling_tools.json'),
    'monarch': os.path.join(current_dir, 'txagent', 'data', 'monarch_tools.json')
}
tool_universe = ToolUniverse(tool_files=default_tool_files, keep_default_tools=False)
tool_universe.load_tools()

# Set ToolRAGModel
RAG_MODEL_NAME = "Qwen/Qwen3-Embedding-8B"
embedding_model = SiliconFlowEmbeddingModel(RAG_MODEL_NAME)
tool_rag_model = ToolRAGModel(embedding_model)
tool_rag_model.load_tool_desc_embedding(tool_universe)

# --- Tool Definition using @tool decorator ---

@tool
def Tool_RAG(query: str, rag_num: int = 5) -> str:
    """
    Retrieves a list of relevant tools for a given query. Use this when you are 
    unsure which specific tool to use or need to discover tools for a task.
    """
    # ... (function implementation is correct) ...
    print(f"\n🔎 RAG searching for tools with query: '{query}'")
    top_tool_names = tool_rag_model.rag_infer(query, top_k=rag_num)
    picked_tools = tool_universe.get_tool_by_name(top_tool_names)
    if not picked_tools:
        return "No relevant tools were found for the query."
    return json.dumps(picked_tools, indent=2)


def load_all_tools() -> List[callable]:
    """
    Loads the RAG tool and dynamically creates all other tools from the ToolUniverse,
    applying the @tool decorator to each one correctly.
    """
    all_tools: List[callable] = [Tool_RAG]
    
    # Map JSON schema types to Python types
    type_map = {"string": str, "integer": int, "number": float, "boolean": bool}

    for tool_schema in tool_universe.all_tools:
        if tool_schema['name'] == 'Tool_RAG':
            continue

        # --- FIX: Dynamically create a Pydantic model for the tool's arguments ---
        
        # 1. Extract parameter details from the schema
        params = tool_schema.get('parameter', {})
        properties = params.get('properties', {})
        required_params = params.get('required', [])
        
        # 2. Define the fields for the Pydantic model
        fields = {}
        for param_name, param_details in properties.items():
            param_type = type_map.get(param_details.get("type"), str)
            description = param_details.get("description", "")
            
            # If the parameter is required, use Ellipsis (...) to mark it as such
            # Otherwise, provide a default value of None to make it optional
            if param_name in required_params:
                fields[param_name] = (param_type, Field(..., description=description))
            else:
                fields[param_name] = (param_type, Field(default=None, description=description))

        # 3. Create the Pydantic model dynamically
        args_schema = create_model(
            tool_schema['name'] + 'Input',
            **fields
        )
        # --- END FIX ---

        def create_tool_from_schema(schema: dict) -> callable:
            def _tool_func(**kwargs):
                # This part remains the same; it correctly passes the received kwargs
                return tool_universe.run_one_function({
                    "name": schema['name'],
                    "arguments": kwargs
                })

            _tool_func.__name__ = schema['name']
            _tool_func.__doc__ = schema['description']
            
            # --- FIX: Pass the dynamically created schema to the decorator ---
            return tool(_tool_func, args_schema=args_schema)
            # --- END FIX ---

        all_tools.append(create_tool_from_schema(tool_schema))
    
    return all_tools

# --- Part 3: Graph Definition and Execution (Refactored) ---

def should_continue(state: AgentState) -> Literal["tools", "__end__"]:
    """Determines whether to continue the loop or end."""
    return "tools" if state["messages"][-1].tool_calls else END

def build_graph(model_name: str) -> CompiledStateGraph:
    """Builds and compiles the LangGraph agent."""
    if "OPENAI_API_KEY" not in os.environ or "GOOGLE_API_KEY" not in os.environ:
        raise ValueError("OPENAI_API_KEY or GOOGLE_API_KEY not found in environment or .env file.")

    # 1. Load all tools and create the ToolNode
    all_tools = load_all_tools()
    tool_node = ToolNode(all_tools)

    # 2. Configure the model to only know about the RAG tool initially
    if "gpt" in model_name.lower():
        model = ChatOpenAI(model=model_name, temperature=0.1, max_tokens=2048)
    elif "gemini" in model_name.lower():
        api_endpoint = os.getenv("GOOGLE_BASE_URL")
        # Initialize the model with the custom endpoint
        model = ChatGoogleGenerativeAI(
            model=model_name,
            client_options={"api_endpoint": api_endpoint}
        )
    else:
        raise ValueError("MODEL_NAME not found in environment or .env file.")
    # The Tool_RAG function itself is now the tool object thanks to the decorator
    model_with_rag_tool = model.bind_tools([Tool_RAG])

    # 3. Define the graph nodes
    def call_model(state: AgentState) -> dict[str, list[BaseMessage]]:
        """Invokes the LLM to get the next action."""
        return {"messages": [model_with_rag_tool.invoke(state["messages"])]}

    # 4. Construct the graph
    workflow = StateGraph(AgentState)
    workflow.add_node("agent", call_model)
    # The ToolNode is used directly as the 'tools' node
    workflow.add_node("tools", tool_node)
    
    workflow.set_entry_point("agent")
    workflow.add_conditional_edges("agent", should_continue)
    workflow.add_edge("tools", "agent")
    
    return workflow.compile()

# --- Part 4: Main Execution Block (Unchanged) ---

if __name__ == "__main__":
    app = build_graph("gemini-2.5-flash")
    example_data ={"id":"ZFU0wbRPwuMG","question_type":"open_ended_multi_choice","question":"What precaution should be taken for patients with a history of allergic disorders before administering Gadavist?","correct_answer":"C","options":{"A":"Administer Gadavist in a diluted form","B":"Avoid Gadavist administration entirely","C":"Assess the patient\u2019s history of reactions to contrast media and ensure trained personnel are available for resuscitation","D":"Perform hemodialysis immediately after administration"}}
    query = json.dumps({"question":example_data["question"],"options":example_data["options"]}, indent=2)
    inputs = {"messages": [HumanMessage(content=query)]}
    
    print(f"🚀 Starting Agent Execution with RAG model: {RAG_MODEL_NAME}...")
    
    for event in app.stream(inputs, stream_mode="values"):
        last_message = event["messages"][-1]
        if isinstance(last_message, AIMessage) and last_message.tool_calls:
            print(f"\n🤖 Agent Thought:\n{last_message.content}")
            print("🛠️ Calling Tools:", last_message.tool_calls)
        elif isinstance(last_message, ToolMessage):
            print(f"\n⚙️ Tool Result ({last_message.name}):\n{last_message.content}")
        elif isinstance(last_message, AIMessage):
            print("\n✅ Final Answer:")
            print(last_message.content)
            
    print("\n🏁 Agent Execution Finished.")