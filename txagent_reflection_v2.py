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
from langchain_core.messages import (AIMessage, BaseMessage, HumanMessage, SystemMessage,
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
    # 'opentarget': os.path.join(current_dir, 'txagent', 'data', 'opentarget_tools.json'),
    'fda_drug_label': os.path.join(current_dir, 'txagent', 'data', 'compressed_tools.json'),
    # 'monarch': os.path.join(current_dir, 'txagent', 'data', 'monarch_tools.json')
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
    return json.dumps(picked_tools) # , indent=2)

import requests
@tool
def FDA_get_all_info_by_drug_name(drug_name):
    """
    Retrieves all information about a given drug name from the FDA.
    drug_name: str
    Returns:
        dict: A dictionary containing all information about the drug.
    """
    params = [
    {
        "base_url": "https://api.fda.gov/drug/label.json",
        "limit": 1,
        "search": f'(openfda.brand_name:("{drug_name}"))'
    },
    {
        "base_url": "https://api.fda.gov/drug/label.json",
        "limit": 1,
        "search": f'(openfda.generic_name:("{drug_name}"))'
    },
    {
        "base_url": "https://api.fda.gov/drug/ndc.json",
        "limit": 1,
        "search": f'(generic_name:("{drug_name}"))'
    },
    {
        "base_url": "https://api.fda.gov/drug/ndc.json",
        "limit": 1,
        "search": f'(brand_name:("{drug_name}"))'
    },
    {
        "base_url": "https://api.fda.gov/drug/label.json",
        "limit": 1,
        "search": f'"{drug_name}"'
    },
    {
        "base_url": "https://api.fda.gov/drug/ndc.json",
        "limit": 1,
        "search": f'"{drug_name}"'
    },
    ]
    
    for param in params:
        base_url = param['base_url']
        param.pop('base_url')
        try:
            # 发送GET请求
            # response = requests.get(base_url, params=param, timeout=10)
            from urllib.parse import urlencode, quote
            query = urlencode(param, quote_via=quote)
            url = f"{base_url}?{query}"
            # print(url)
            response = requests.get(url, timeout=10)
            # 检查请求是否成功
            response.raise_for_status()
            
            # 返回JSON格式的响应内容
            result = response.json()
            print("获取成功")
            # result['entity'] = drug_name  # 添加药品名称到结果中
            return result
            
        except requests.exceptions.RequestException as e:
            # print(f"获取 {drug_name} 信息时发生错误: {e}")
            pass
    return None

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
        print(f"-> Capturing tool output::")
        print(f"-> Func is: {func}")
        print(f"-> Args is: {args}")
        print(f"-> Kwargs is: {kwargs}")
        with contextlib.redirect_stdout(output_buffer):
            result = func(*args, **kwargs)
        # print(f"Tool output is: {result}")
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
            print(f"Error message is: {error_message}")
            return f"{REFLECTION_TRIGGER}: {error_message}"

        # If successful, format and return the result
        result_str = json.dumps(result) if not isinstance(result, str) else result
        print(f"Result string is: {result_str}")
        if captured_output:
            return f"Tool Logs:\n{captured_output}\n\nTool Result:\n{result_str}"
        else:
            return result_str

    except Exception as e:
        print(f"Exception is: {e}")
        return f"{REFLECTION_TRIGGER}: An exception occurred during tool execution: {e}"

def load_all_tools_map() -> Dict[str, Callable]:
    all_tools_map = {"Tool_RAG": Tool_RAG, "FDA_get_all_info_by_drug_name": FDA_get_all_info_by_drug_name}
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
        if tool_schema['name'] == 'Tool_RAG' or tool_schema['name'] == 'FDA_get_all_info_by_drug_name':
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
    Determines the next step after tool execution.
    - If Tool_RAG was used, route to tool_manager.
    - If ALL tools failed, route to reflection.
    - If at least one tool succeeded, route back to the agent.
    """
    print(f"-> After tool execution::")
    last_message = state["messages"][-1]
    
    # Handle the specific case of the Tool_RAG tool
    if isinstance(last_message, ToolMessage) and last_message.name == "Tool_RAG":
        print("-> RAG tool was used. Routing to tool_manager.")
        return "tool_manager"
    
    # Find the last AI message to determine how many tools were called
    last_ai_message_index = -1
    for i in range(len(state["messages"]) - 1, -1, -1):
        if isinstance(state["messages"][i], AIMessage) and state["messages"][i].tool_calls:
            last_ai_message_index = i
            break
            
    # Proceed only if we found an AI message with tool calls
    if last_ai_message_index != -1:
        # Get the number of tool calls and their corresponding results
        num_tool_calls = len(state["messages"][last_ai_message_index].tool_calls)
        recent_tool_messages = state["messages"][last_ai_message_index + 1 : last_ai_message_index + 1 + num_tool_calls]
        
        # Count how many of the recent tool calls failed
        failure_count = 0
        for msg in recent_tool_messages:
            if isinstance(msg, ToolMessage) and "TOOL_REFLECTION_NEEDED" in msg.content:
                failure_count += 1
        
        print(f"-> Tool execution summary: {failure_count} failed out of {num_tool_calls} total.")

        # If the number of failures equals the number of calls, route to reflection
        if num_tool_calls > 0 and failure_count == num_tool_calls:
            print("-> All tools failed. Routing to reflection node.")
            return "reflection"
                
    # In all other cases (all tools succeeded or partial success), route to the agent
    print("-> At least one tool succeeded. Routing back to agent.")
    return "agent"
    
# --- Part 3: Graph Definition and Execution (MODIFIED) ---
def should_continue(state: AgentState) -> Literal["tools", "__end__"]:
    print("-> Checking if should continue...")
    return "tools" if state["messages"][-1].tool_calls else END

def build_graph(model_name: str) -> CompiledStateGraph:
    all_tools_map = load_all_tools_map()
    model = ChatOpenAI(model = model_name, temperature = 0.1,
                        base_url = os.getenv("OPENAI_BASE_URL"),api_key = os.getenv("OPENAI_API_KEY"))

    def call_agent(state: AgentState) -> dict:
        print("\n🤔 Agent thinking...")
        current_tools = state.get("available_tools", [Tool_RAG, FDA_get_all_info_by_drug_name])
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

    def custom_tool_executor(state: AgentState) -> dict:
        """
        A custom replacement for ToolNode that executes tool calls and returns ToolMessages.
        """
        print("\n🛠️ Executing tools with custom executor...")
        
        # 1. Get the last message, which should be the AI's request to call tools
        last_message = state['messages'][-1]
        if not hasattr(last_message, 'tool_calls') or not last_message.tool_calls:
            print(" -> No tool calls found in the last message.")
            return {}

        # This map needs to be accessible to the function.
        # Since we define this inside build_graph, it will have access via closure.
        tool_map = load_all_tools_map() 
        
        # 2. Iterate through each tool call and execute it
        tool_messages = []
        for tool_call in last_message.tool_calls:
            tool_name = tool_call['name']
            tool_args = tool_call['args']
            tool_call_id = tool_call['id']
            
            print(f"  -> Calling tool '{tool_name}' with args: {tool_args}")

            # 3. Find and run the corresponding tool function
            if tool_name in tool_map:
                tool_function = tool_map[tool_name]
                try:
                    # The actual tool execution happens here
                    result = tool_function.invoke(tool_args)
                    
                    # 4. Create a ToolMessage with the result
                    tool_messages.append(ToolMessage(content=str(result), tool_call_id=tool_call_id))
                except Exception as e:
                    error_message = f"Error executing tool {tool_name}: {e}"
                    print(f"  -> {error_message}")
                    tool_messages.append(ToolMessage(content=error_message, tool_call_id=tool_call_id))
            else:
                not_found_message = f"Tool '{tool_name}' not found."
                print(f"  -> {not_found_message}")
                tool_messages.append(ToolMessage(content=not_found_message, tool_call_id=tool_call_id))

        # 5. Return the results to be added to the state
        return {"messages": tool_messages}

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
    workflow.add_node("tools", custom_tool_executor)
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
    model_name = os.getenv("MODEL_NAME")
    app = build_graph(model_name)
    initial_tools = [Tool_RAG, FDA_get_all_info_by_drug_name]
    example_data ={"id":"ZFU0wbRPwuMG","question_type":"open_ended_multi_choice","question":"What precaution should be taken for patients with a history of allergic disorders before administering Gadavist?","correct_answer":"C","options":{"A":"Administer Gadavist in a diluted form","B":"Avoid Gadavist administration entirely","C":"Assess the patient\u2019s history of reactions to contrast media and ensure trained personnel are available for resuscitation","D":"Perform hemodialysis immediately after administration"}}
    query = json.dumps({"question":example_data["question"],"options":example_data["options"]}, indent=2)
    config = {"configurable": {"thread_id": "test-thread"},
              "recursion_limit": 100}
    with open(os.path.join(current_dir, 'txagent', 'data', 'system_prompt.md'), 'r') as file:
        system_prompt = file.read()
    inputs = {
        "messages": [
            SystemMessage(content=system_prompt),
            HumanMessage(content=query)
            ],
        "available_tools": initial_tools
    }
    print(f"🚀 Starting Agent Execution. Initial tool: {[t.name for t in initial_tools]}")

    final_state = app.invoke(inputs, config=config)
    print(f"Final state is: {final_state}")

    # for event in app.stream(inputs, config=config, stream_mode="values"):
    #     last_message = event["messages"][-1]
    #     # print("Event is:", type(event), "Length is:", len(event["messages"]))
    #     # print("Last message is:", type(last_message), "Content is:", last_message.content)
    #     if isinstance(last_message, AIMessage) and last_message.tool_calls:
    #         print(f"\n🤖 Agent Thought:\n{last_message.content}")
    #         print("🛠️ Calling Tools:", last_message.tool_calls)
    #     elif isinstance(last_message, ToolMessage):
    #         print(f"\n⚙️ Tool Result ({last_message.name}):\n{last_message.content[:100]}")
    #     elif isinstance(last_message, AIMessage):
    #         print("\n✅ Final Answer:")
    #         print(last_message.content)
            
    print("\n🏁 Agent Execution Finished.")