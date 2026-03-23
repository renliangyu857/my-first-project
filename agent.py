import os, json, time, operator, sqlite3
from typing import List, Annotated, Union
from typing_extensions import TypedDict
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.sqlite import SqliteSaver 
from tools import arxiv_research_tool, query_research_db, summarize_paper_tool

class AgentState(TypedDict):
    input: str
    plan: List[str]
    past_steps: Annotated[List[str], operator.add]
    messages: Annotated[List[Union[AIMessage, HumanMessage, ToolMessage]], operator.add]
    metrics: Annotated[dict, operator.ior] 

def create_research_agent():
    api_key, base_url = os.getenv("OPENAI_API_KEY"), os.getenv("OPENAI_API_BASE")
    planner_llm = ChatOpenAI(model="Qwen/Qwen2.5-72B-Instruct", temperature=0.1, openai_api_key=api_key, base_url=base_url)
    executor_llm = ChatOpenAI(model="Qwen/Qwen2.5-7B-Instruct", temperature=0.1, openai_api_key=api_key, base_url=base_url)
    tools = [arxiv_research_tool, query_research_db, summarize_paper_tool]

    def planner_node(state: AgentState):
        start = time.time()
        prompt = ChatPromptTemplate.from_messages([
            ("system", "你是一个科研主管。请记住用户的个人信息（如学校、专业等）。参考历史对话，将需求拆解为2-4步任务。"),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}")
        ])
        # 💡 增强点：增加记忆窗口到 10 条
        history = state["messages"][-10:] if state["messages"] else []
        res = (prompt | planner_llm).invoke({"input": state["input"], "history": history})
        
        # 💡 增强点：更稳健的任务提取
        lines = res.content.split("\n")
        tasks = [l.strip() for l in lines if any(l.strip().startswith(str(i)) for i in range(1, 10)) or l.strip().startswith("-")]
        return {"plan": tasks, "metrics": {"01_Planner_Logic": round(time.time()-start, 2)}}

    def executor_node(state: AgentState):
        start = time.time()
        task = state["plan"][0]
        facts = "\n".join(state["past_steps"])
        # 💡 增强点：让执行器也知道用户是谁
        sys_prompt = f"任务：{task}\n已知事实库：{facts}\n请调用工具，注意结合上下文中的用户信息。"
        is_complex = any(kw in task for kw in ["对比", "分析", "总结"])
        llm = (planner_llm if is_complex else executor_llm).bind_tools(tools)
        res = llm.invoke([SystemMessage(content=sys_prompt), HumanMessage(content=task)])
        return {"messages": [res], "metrics": {"02_Executor_Thought": round(time.time()-start, 2)}}

    def synthesizer_node(state: AgentState):
        start = time.time()
        all_facts = "\n".join(state["past_steps"])
        
        # 💡 增强点：提取最近对话，确保“我是什么学校的”能被看见
        history_context = ""
        for m in state["messages"][-8:]:
            role = "用户" if isinstance(m, HumanMessage) else "助手"
            history_context += f"{role}: {m.content}\n"

        res = planner_llm.invoke([
            SystemMessage(content="你是首席科学家。请结合历史对话中的用户信息（如学校）和科研事实撰写报告。"),
            HumanMessage(content=f"--- 历史对话回溯 ---\n{history_context}\n\n--- 科研事实积累 ---\n{all_facts}\n\n当前用户需求：{state['input']}")
        ])
        return {"messages": [res], "metrics": {"04_Synthesizer_Final": round(time.time()-start, 2)}}

    # 其他节点保持原样
    timed_tool_node = lambda state: {**ToolNode(tools).invoke(state), "metrics": {"03_Tools_IO": 0.1}} 
    observer_node = lambda state: {"past_steps": [str(state["messages"][-1].content)[:500]] if isinstance(state["messages"][-1], ToolMessage) else [], "plan": state["plan"][1:]}

    workflow = StateGraph(AgentState)
    workflow.add_node("planner", planner_node); workflow.add_node("executor", executor_node)
    workflow.add_node("tools", timed_tool_node); workflow.add_node("observer", observer_node)
    workflow.add_node("synthesizer", synthesizer_node)
    
    workflow.add_edge(START, "planner")
    workflow.add_conditional_edges("planner", lambda x: "executor" if x["plan"] else "synthesizer")
    workflow.add_conditional_edges("executor", lambda x: "tools" if x["messages"][-1].tool_calls else "observer")
    workflow.add_edge("tools", "observer")
    workflow.add_conditional_edges("observer", lambda x: "executor" if x["plan"] else "synthesizer")
    workflow.add_edge("synthesizer", END)

    conn = sqlite3.connect("agent_memory.db", check_same_thread=False)
    return workflow.compile(checkpointer=SqliteSaver(conn))