# agent_main_v7_with_reflection.py
"""
This version introduces a reflection mechanism to handle tool failures.
When a tool returns an error or no data, the agent enters a 'reflection'
state where it receives guidance on how to retry, making it more resilient.
"""
import io
import contextlib
import json
import operator
import os
from typing import Annotated, Dict, List, Literal, TypedDict, Callable, Type

from dotenv import load_dotenv
from langchain_core.messages import (AIMessage, BaseMessage, HumanMessage,
                                     ToolMessage)
from langchain_core.tools import tool
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import Field, create_model, BaseModel

# --- Import the external ToolRAGModel package ---
from txagent.toolrag import SiliconFlowEmbeddingModel, ToolRAGModel
from tooluniverse import ToolUniverse

# --- Part 1: State Definition (Unchanged) ---
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    available_tools: List[Callable]

# --- Part 2: Tools Abstraction (MODIFIED) ---

# --- Global Instantiation and Initialization (Unchanged) ---
load_dotenv(override=True)
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

@tool
def Tool_RAG(query: str, rag_num: int = 5) -> str:
    """
    Retrieves a list of relevant tools for a given query. Use this when you are
    unsure which specific tool to use or need to discover tools for a task.
    This should be the first tool you use in a complex workflow.
    """
    print(f"\n🔎 RAG searching for tools with query: '{query}'")
    top_tool_names = tool_rag_model.rag_infer(query, top_k=rag_num)
    picked_tools = tool_universe.get_tool_by_name(top_tool_names)
    if not picked_tools:
        return "No relevant tools were found for the query."
    return json.dumps(picked_tools, indent=2)

### --- MODIFICATION: Enhanced error detection in tool output capture --- ###
def capture_tool_output(func, *args, **kwargs) -> str:
    """
    Executes a function and captures its output. It now detects more failure
    conditions (like empty results) and returns a specific trigger string to
    initiate a reflection step in the graph.
    """
    output_buffer = io.StringIO()
    result = None
    REFLECTION_TRIGGER = "TOOL_REFLECTION_NEEDED"

    try:
        with contextlib.redirect_stdout(output_buffer):
            result = func(*args, **kwargs)
        
        captured_output = output_buffer.getvalue().strip()
        is_error = False
        error_message = ""

        # 1. Check for explicit 'NOT_FOUND' error from the tool_universe
        if isinstance(result, str) and 'NOT_FOUND' in result:
            is_error = True
            error_message = (
                "The tool returned a 'NOT_FOUND' error. This means the resource "
                "you searched for (e.g., the drug_name) does not exist in the database. "
                "Please verify the spelling or try a different query."
            )
        # 2. Check for empty or null-like results, which indicate failure to find data
        elif result is None or result == [] or result == {}:
            is_error = True
            error_message = (
                "The tool executed successfully but returned no data. This could mean "
                "the requested information does not exist for the given input. "
                "Consider trying a different, more general tool."
            )

        if is_error:
            return f"{REFLECTION_TRIGGER}: {error_message}"

        # If successful, format and return the result
        result_str = json.dumps(result) if not isinstance(result, str) else result
        if captured_output:
            return f"Tool Logs:\n{captured_output}\n\nTool Result:\n{result_str}"
        else:
            return result_str

    except Exception as e:
        return f"{REFLECTION_TRIGGER}: An exception occurred during tool execution: {e}"

def load_all_tools_map() -> Dict[str, Callable]:
    all_tools_map = {"Tool_RAG": Tool_RAG}
    type_map = {"string": str, "integer": int, "number": float, "boolean": bool}

    def create_wrapped_tool(schema: dict, args_schema: Type[BaseModel]) -> Callable:
        def _tool_func(**kwargs):
            tool_call_args = {
                "name": schema['name'],
                "arguments": kwargs
            }
            return capture_tool_output(tool_universe.run_one_function, tool_call_args)

        _tool_func.__name__ = schema['name']
        _tool_func.__doc__ = schema['description']
        return tool(_tool_func, args_schema=args_schema)

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

        args_schema_model = create_model(tool_schema['name'] + 'Input', **fields)
        tool_func = create_wrapped_tool(tool_schema, args_schema_model)
        all_tools_map[tool_schema['name']] = tool_func
        
    return all_tools_map

### --- MODIFICATION: Updated conditional routing to include reflection --- ###
def after_tool_execution(state: AgentState) -> Literal["tool_manager", "agent", "reflection"]:
    """
    Determines the next step after a tool has been executed.
    - If Tool_RAG was used, route to the tool_manager.
    - If a tool failed (indicated by a specific trigger), route to reflection.
    - Otherwise, route back to the agent to process the output.
    """
    last_message = state["messages"][-1]
    if isinstance(last_message, ToolMessage):
        if last_message.name == "Tool_RAG":
            print("-> RAG tool was used. Routing to tool_manager.")
            return "tool_manager"
        # Check for the reflection trigger in the tool's output content
        if "TOOL_REFLECTION_NEEDED" in last_message.content:
            print("-> Tool failed. Routing to reflection node.")
            return "reflection"
            
    print("-> A standard tool was used successfully. Routing back to agent.")
    return "agent"
    
# --- Part 3: Graph Definition and Execution (MODIFIED) ---
def should_continue(state: AgentState) -> Literal["tools", "__end__"]:
    return "tools" if state["messages"][-1].tool_calls else END

def build_graph(model_name: str) -> CompiledStateGraph:
    all_tools_map = load_all_tools_map()
    all_tools_list = list(all_tools_map.values())
    tool_node = ToolNode(all_tools_list)

    if "gpt" in model_name.lower():
        model = ChatOpenAI(model=model_name, temperature=0.1, max_tokens=2048)
    elif "gemini" in model_name.lower():
        api_endpoint = os.getenv("GOOGLE_BASE_URL")
        model = ChatGoogleGenerativeAI(
            model=model_name,
            client_options={"api_endpoint": api_endpoint}
        )
    else:
        raise ValueError(f"Unsupported model name: {model_name}")

    def call_agent(state: AgentState) -> dict:
        print("\n🤔 Agent thinking...")
        current_tools = state.get("available_tools", [Tool_RAG])
        model_with_current_tools = model.bind_tools(current_tools)
        response = model_with_current_tools.invoke(state["messages"])
        return {"messages": [response]}

    def manage_tools(state: AgentState) -> dict:
        last_message = state["messages"][-1]
        if not isinstance(last_message, ToolMessage) or last_message.name != "Tool_RAG":
            return {}

        print("\n🛠️ Managing discovered tools...")
        try:
            discovered_tools_json = json.loads(last_message.content)
            discovered_tool_names = [t['name'] for t in discovered_tools_json]
            newly_available_tools = [all_tools_map[name] for name in discovered_tool_names if name in all_tools_map]
            
            if Tool_RAG not in newly_available_tools:
                newly_available_tools.insert(0, Tool_RAG)

            print(f"   -> Made {len(newly_available_tools)} tools available: {[t.name for t in newly_available_tools]}")
            
            guidance_message = HumanMessage(
                content=(
                    "I have retrieved the following tools based on your request: "
                    f"{', '.join(discovered_tool_names)}. Please use these tools now to "
                    "answer the original user query."
                )
            )
            return {"available_tools": newly_available_tools, "messages": [guidance_message]}
        except (json.JSONDecodeError, KeyError) as e:
            print(f"   -> Error parsing Tool_RAG output: {e}")
            error_message = HumanMessage(content=f"Error processing discovered tools: {last_message.content}")
            return {"messages": [error_message]}

    ### --- MODIFICATION: Added a new node for reflection --- ###
    def generate_reflection(state: AgentState) -> dict:
        """
        Generates a reflection message for the agent when a tool fails.
        This provides guidance based on the suggestions provided.
        """
        print("\n🤔 Generating reflection on tool failure...")
        last_message = state["messages"][-1]
        if not isinstance(last_message, ToolMessage):
            return {}

        error_details = last_message.content.replace("TOOL_REFLECTION_NEEDED:", "").strip()
        tool_name = last_message.name

        reflection_message_content = f"""The tool '{tool_name}' failed or returned no data.
Reason: {error_details}

Here are some suggestions to fix this:
1.  **Refine Keywords**: The keywords you used (e.g., drug name) might not be accurate. Please verify them.
2.  **Choose a Different Tool**: The tool '{tool_name}' might not be suitable for your request, as some data fields don't exist in all records. Using a different tool might help.

**Action**: Please analyze the problem again. You can use `Tool_RAG` to find a more suitable tool, or try a different tool with different arguments.
"""
        reflection_message = HumanMessage(content=reflection_message_content)
        return {"messages": [reflection_message]}

    workflow = StateGraph(AgentState)
    workflow.add_node("agent", call_agent)
    workflow.add_node("tools", tool_node)
    workflow.add_node("tool_manager", manage_tools)
    # --- MODIFICATION START ---
    # Add the new reflection node to the graph
    workflow.add_node("reflection", generate_reflection)
    # --- MODIFICATION END ---
    
    workflow.set_entry_point("agent")
    
    workflow.add_conditional_edges("agent", should_continue, {"tools": "tools", "__end__": END})
    
    ### --- MODIFICATION: Re-wired the graph to include the reflection loop --- ###
    # This new conditional edge from 'tools' directs flow based on the outcome
    workflow.add_conditional_edges(
        "tools",
        after_tool_execution,
        {
            "tool_manager": "tool_manager",
            "reflection": "reflection", # New path for handling tool failures
            "agent": "agent"
        }
    )
    # The tool_manager and reflection nodes both route back to the agent
    workflow.add_edge("tool_manager", "agent")
    workflow.add_edge("reflection", "agent")
    ### --- MODIFICATION END ---
    
    return workflow.compile()

# --- Part 4: Main Execution Block (Unchanged) ---
if __name__ == "__main__":
    app = build_graph("gpt-4o-mini")
    initial_tools = [Tool_RAG]
    example_data ={"id":"ZFU0wbRPwuMG","question_type":"open_ended_multi_choice","question":"What precaution should be taken for patients with a history of allergic disorders before administering Gadavist?","correct_answer":"C","options":{"A":"Administer Gadavist in a diluted form","B":"Avoid Gadavist administration entirely","C":"Assess the patient\u2019s history of reactions to contrast media and ensure trained personnel are available for resuscitation","D":"Perform hemodialysis immediately after administration"}}
    query = json.dumps({"question":example_data["question"],"options":example_data["options"]}, indent=2)
    config = {"configurable": {"thread_id": "test-thread"}}
    inputs = {
        "messages": [HumanMessage(content=query)],
        "available_tools": initial_tools
    }

    print(f"🚀 Starting Agent Execution. Initial tool: {[t.name for t in initial_tools]}")

    for event in app.stream(inputs, config=config, stream_mode="values"):
        last_message = event["messages"][-1]
        # print("Event is:", type(event), "Length is:", len(event["messages"]))
        # print("Last message is:", type(last_message), "Content is:", last_message.content)
        if isinstance(last_message, AIMessage) and last_message.tool_calls:
            print(f"\n🤖 Agent Thought:\n{last_message.content}")
            print("🛠️ Calling Tools:", last_message.tool_calls)
        elif isinstance(last_message, ToolMessage):
            print(f"\n⚙️ Tool Result ({last_message.name}):\n{last_message.content}")
        elif isinstance(last_message, AIMessage):
            print("\n✅ Final Answer:")
            print(last_message.content)
            
    print("\n🏁 Agent Execution Finished.")