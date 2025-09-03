# api_server.py

import os
import uvicorn
import logging
from typing import List, Dict, Any, Optional, Literal, Union
from contextlib import asynccontextmanager

# --- FastAPI 和 Pydantic 框架导入 ---
from fastapi import FastAPI, HTTPException, Request, Depends
from pydantic import BaseModel, ConfigDict, model_validator

# --- LangChain 消息类导入 (仅用于数据格式转换) ---
from langchain_core.messages import (AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage)

# --- 从 graph.py 模块导入代理构建函数和配置 ---
# from graph import build_graph, RAG_MODEL_NAME
from txagent_reflection_v2 import build_graph, RAG_MODEL_NAME


async def log_request_body(request: Request):
    """一个 FastAPI 依赖项，用于记录传入POST请求的请求体。"""
    try:
        body = await request.json()
        logger.info(f"收到的请求体: {body}")
    except Exception:
        logger.warning("无法解析请求体。")

# ==============================================================================
# --- 0. 日志配置 ---
# ==============================================================================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ==============================================================================
# --- 1. 标准化的 API 数据结构 ---
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
            if self.text is None: raise ValueError("对于 'text' 类型的块, 'text' 字段是必需的。")
            if self.source_type or self.data or self.mime_type or self.url: raise ValueError("对于 'text' 类型的块, 只应提供 'text' 字段。")
        # 当前代理只支持文本，因此不实现多模态的检查。
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
# --- 2. FastAPI 应用设置 ---
# ==============================================================================

# --- Part 2.1: 使用 Lifespan 管理器在应用启动时加载模型和图 ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI 的生命周期管理器。在服务器启动时执行加载操作，在关闭时执行清理操作。
    """
    # 启动时: 从 graph.py 构建 LangGraph 代理，并将其存储在 app state 中
    logger.info("FastAPI 应用启动: 正在构建 LangGraph 代理...")
    model_name = os.getenv("MODEL_NAME", "gemini-1.5-flash") # 默认为 gemini
    app.state.graph = build_graph(model_name)
    logger.info(f"使用模型 '{model_name}' 的 LangGraph 代理已成功构建。")
    yield
    # 关闭时: 如果需要，可以在此清理资源
    logger.info("FastAPI 应用已关闭。")

app = FastAPI(
    title="LangGraph RAG Agent API",
    description="一个为具备动态工具检索能力的 LangGraph 代理提供的 API 服务器。",
    version="1.0.0",
    lifespan=lifespan  # 注册生命周期管理器
)

# --- Part 2.2: 消息格式转换的辅助函数 ---
def convert_api_messages_to_langchain(messages: List[Message]) -> List[BaseMessage]:
    """将 API 消息格式转换为 LangChain 的 BaseMessage 格式。"""
    lc_messages = []
    for msg in messages:
        lc_content = None
        if isinstance(msg.content, str):
            lc_content = msg.content
        elif isinstance(msg.content, list): 
            lc_content = [block.model_dump() for block in msg.content]
        else:
            raise HTTPException(status_code=400, detail="无效的消息内容类型。")
        
        if msg.role == "user":
            lc_messages.append(HumanMessage(content=lc_content))
        elif msg.role == "assistant":
            lc_messages.append(AIMessage(content=lc_content))
        elif msg.role == "system":
            lc_messages.append(SystemMessage(content=lc_content))
    return lc_messages

def convert_langchain_messages_to_api(messages: List[BaseMessage]) -> List[Message]:
    """将 LangChain 的 BaseMessage 格式转换回 API 消息格式。"""
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
            continue # 跳过未知的消息类型
        
        api_messages.append(Message(role=role, content=str(msg.content)))
    return api_messages


# --- Part 2.3: API 端点 (Endpoints) ---
@app.post("/invoke", response_model=AgentResponse) #, dependencies=[Depends(log_request_body)])
async def invoke_agent(agent_request: AgentRequest, http_request: Request):
    """
    接收请求，调用 LangGraph 代理，并返回最终响应。
    """
    logger.info(f"收到请求，request_id: {agent_request.request_id}")
    
    # 1. 从应用 state 中获取已编译的图
    graph = http_request.app.state.graph
    if not graph:
        raise HTTPException(status_code=503, detail="代理图尚未初始化，请稍后再试。")

    # 2. 将传入消息转换为 LangChain 格式
    try:
        input_messages = convert_api_messages_to_langchain(agent_request.messages)
    except HTTPException as e:
        raise e
    
    # 3. 检查并应用元数据中的系统提示 (System Prompt)
    if agent_request.metadata and "system_prompt" in agent_request.metadata:
        system_prompt = agent_request.metadata["system_prompt"]
        if isinstance(system_prompt, str) and system_prompt:
            logger.info("正在应用元数据中的系统提示。")
            system_message = SystemMessage(content=system_prompt)
            input_messages.insert(0, system_message) # 将系统提示插入到消息列表的开头
    else:
        # 使用local system prompt
        logger.info("正在应用本地系统提示。")
        with open(os.path.join(os.path.dirname(__file__), 'txagent', 'data', 'system_prompt.md'), 'r') as file:
            system_prompt = file.read()
        system_message = SystemMessage(content=system_prompt)
        input_messages.insert(0, system_message)
    
    # 4. 准备图的输入
    inputs = {"messages": input_messages}
    
    # 5. 异步调用图并获取最终状态
    try:
        # ================================================================= #
        # === 核心修改：使用 await 和 ainvoke 来进行非阻塞的异步调用 === #
        # ================================================================= #
        final_state = await graph.ainvoke(inputs, {"recursion_limit": 80})
        
        # 6. 提取最终答案和完整的消息历史
        final_messages_lc = final_state.get("messages", [])
        if not final_messages_lc or not isinstance(final_messages_lc[-1], AIMessage):
             raise HTTPException(status_code=500, detail="代理未能生成最终答案。")
            
        final_answer = final_messages_lc[-1].content
        complete_messages_api = convert_langchain_messages_to_api(final_messages_lc)
        
        logger.info(f"成功处理请求，request_id: {agent_request.request_id}")
        
        # 7. 返回结构化的响应
        return AgentResponse(
            request_id=agent_request.request_id,
            answer=final_answer,
            complete_messages=complete_messages_api
        )
    except Exception as e:
        logger.error(f"为 request_id {agent_request.request_id} 调用图时出错: {e}")
        raise HTTPException(status_code=500, detail=f"代理执行期间发生错误: {e}")

@app.get("/", tags=["Health Check"])
def health_check():
    """
    提供一个简单的健康检查端点，以确认 API 正在运行并查看模型信息。
    """
    model_name = os.getenv("MODEL_NAME", "gemini-1.5-flash")
    return {
        "status": "ok",
        "message": "LangGraph RAG Agent API 正在运行。",
        "details": {
            "active_llm": model_name,
            "rag_embedding_model": RAG_MODEL_NAME  # 从 graph 模块导入
        }
    }

# ==============================================================================
# --- 3. 主程序执行块 ---
# ==============================================================================
if __name__ == "__main__":
    # 运行此服务器的推荐方式:
    # 1. 确保你有一个 .env 文件，其中包含 GOOGLE_API_KEY 或 OPENAI_API_KEY。
    # 2. 在终端中运行以下命令以启动支持多进程的服务器:
    #    uvicorn api_server:app --host 0.0.0.0 --port 8128 --workers 4
    #
    #    - '--workers 4' 会启动4个进程来处理请求，实现真正的并行。
    #    - 可以根据你的CPU核心数调整 worker 数量。
    
    # 以下代码用于本地快速调试，但它是单进程的。
    print("正在以单进程开发模式启动服务器。")
    print("为了获得最佳性能和并发能力，请使用 'uvicorn api_server:app --workers 4' 命令启动。")
    uvicorn.run("api_server:app", host="127.0.0.1", port=8128, reload=True)