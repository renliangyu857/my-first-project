import os, json, time, operator, sqlite3
from typing import List, Annotated, Union, Literal
from typing_extensions import TypedDict

# LangChain 核心
from langchain_core.globals import set_llm_cache
from langchain_core.caches import InMemoryCache
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.sqlite import SqliteSaver # 🚀 引入持久化数据库

from tools import arxiv_research_tool, query_research_db, summarize_paper_tool

set_llm_cache(InMemoryCache())

# 1. 增强型状态定义
class AgentState(TypedDict):
    input: str                         # 当前输入
    plan: List[str]                    # 任务栈
    past_steps: Annotated[List[str], operator.add] # 跨轮次的科研事实积累
    messages: Annotated[List[Union[AIMessage, HumanMessage, ToolMessage]], operator.add] # 完整对话历史
    metrics: Annotated[dict, operator.ior] # 性能量化

def create_research_agent():
    api_key, base_url = os.getenv("OPENAI_API_KEY"), os.getenv("OPENAI_API_BASE")
    planner_llm = ChatOpenAI(model="Qwen/Qwen2.5-72B-Instruct", temperature=0.1, openai_api_key=api_key, base_url=base_url)
    executor_llm = ChatOpenAI(model="Qwen/Qwen2.5-7B-Instruct", temperature=0.1, openai_api_key=api_key, base_url=base_url)
    tools = [arxiv_research_tool, query_research_db, summarize_paper_tool]

    # --- 节点 1: Planner (带历史感知的规划器) ---
    def planner_node(state: AgentState):
        start = time.time()
        # 🧠 核心改进：让 Planner 看到历史对话，实现“记忆”
        prompt = ChatPromptTemplate.from_messages([
            ("system", "你是一个科研主管。参考历史对话，将当前需求拆解为2-4步。只输出任务列表。"),
            MessagesPlaceholder(variable_name="history"), # 注入对话历史
            ("human", "{input}")
        ])
        
        # 提取最近 5 条历史
        history = state["messages"][-5:] if state["messages"] else []
        res = (prompt | planner_llm).invoke({"input": state["input"], "history": history})
        tasks = [t.strip() for t in res.content.split("\n") if t.strip() and (t[0].isdigit() or t.startswith("-"))]
        return {"plan": tasks, "metrics": {"Planner": round(time.time()-start, 2)}}

    # --- 节点 2: Executor (计时加速版) ---
    def executor_node(state: AgentState):
        start = time.time()
        task = state["plan"][0]
        # 汇总所有轮次的事实
        facts = "\n".join(state["past_steps"])
        
        sys_prompt = f"任务：{task}\n已知事实库：{facts}\n请调用工具。若信息已足够请回复‘完成’。"
        
        # 智能分流：如果是复杂论文对比，直接调 72B 并发
        is_complex = any(kw in task for kw in ["对比", "分析", "两篇", "多篇"])
        llm = (planner_llm if is_complex else executor_llm).bind_tools(tools, parallel_tool_calls=is_complex)
        
        res = llm.invoke([SystemMessage(content=sys_prompt), HumanMessage(content=task)])
        return {"messages": [res], "metrics": {"Executor": round(time.time()-start, 2)}}

    # --- 节点 3: Observer (物理削减任务栈) ---
    def observer_node(state: AgentState):
        last_msg = state["messages"][-1]
        if isinstance(last_msg, ToolMessage):
            # 将新事实存入跨轮次记忆库
            return {"past_steps": [f"发现: {str(last_msg.content)[:800]}"], "plan": state["plan"][1:]}
        return {"plan": state["plan"][1:]} if not last_msg.tool_calls else {}

    # --- 节点 4: Synthesizer (全景报告) ---
    def synthesizer_node(state: AgentState):
        start = time.time()
        # 汇总所有科研积累
        all_facts = "\n".join(state["past_steps"])
        res = planner_llm.invoke([
            SystemMessage(content="你是首席科学家，根据所有执行记录写一份深度报告。"),
            HumanMessage(content=f"历史科研积累：\n{all_facts}\n\n当前用户需求：{state['input']}")
        ])
        return {"messages": [res], "metrics": {"Synthesizer": round(time.time()-start, 2)}}

    # 🔗 图网络拓扑修正
    workflow = StateGraph(AgentState)
    workflow.add_node("planner", planner_node)
    workflow.add_node("executor", executor_node)
    workflow.add_node("tools", ToolNode(tools))
    workflow.add_node("observer", observer_node)
    workflow.add_node("synthesizer", synthesizer_node)

    workflow.add_edge(START, "planner")
    workflow.add_conditional_edges("planner", lambda x: "executor" if x["plan"] else "synthesizer")
    workflow.add_conditional_edges("executor", lambda x: "tools" if x["messages"][-1].tool_calls else "observer")
    workflow.add_edge("tools", "observer")
    workflow.add_conditional_edges("observer", lambda x: "executor" if x["plan"] else "synthesizer")
    workflow.add_edge("synthesizer", END)

    # 🚀 持久化记忆：使用 Sqlite 数据库
    conn = sqlite3.connect("agent_memory.db", check_same_thread=False)
    memory = SqliteSaver(conn)
    return workflow.compile(checkpointer=memory)