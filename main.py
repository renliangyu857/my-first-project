from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List
import uuid
import time
from agent import create_research_agent
from langchain_core.messages import HumanMessage, AIMessage

app = FastAPI(title="DeepResearch API Server", version="1.0.0")

# 全局初始化 Agent 引擎
agent_executor = create_research_agent()

# 定义请求数据结构
class ResearchRequest(BaseModel):
    query: str
    thread_id: Optional[str] = None

# 定义响应数据结构
class ResearchResponse(BaseModel):
    thread_id: str
    answer: str
    metrics: dict
    status: str

@app.get("/")
def read_root():
    return {"status": "Online", "service": "DeepResearch AI Engine"}

@app.post("/v1/chat", response_model=ResearchResponse)
async def chat_with_agent(req: ResearchRequest):
    # 如果没传 ID，就生成一个新的
    current_thread_id = req.thread_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": current_thread_id}}
    
    try:
        # 执行 Agent 逻辑
        # 注意：在 API 环境中，我们通常直接获取最终结果，或者通过 WebSocket 推送流式数据
        result = agent_executor.invoke(
            {"input": req.query, "messages": [HumanMessage(content=req.query)], "metrics": {}},
            config=config
        )
        
        # 提取最终回答
        final_answer = next((m.content for m in reversed(result["messages"]) 
                             if isinstance(m, AIMessage) and not m.tool_calls), "分析完成")
        
        return ResearchResponse(
            thread_id=current_thread_id,
            answer=final_answer,
            metrics=result.get("metrics", {}),
            status="success"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)