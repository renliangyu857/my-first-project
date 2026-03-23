import streamlit as st
import uuid, time, os, pandas as pd
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from agent import create_research_agent

load_dotenv()
st.set_page_config(page_title="科研加速引擎", layout="wide")

if "agent_executor" not in st.session_state:
    st.session_state.agent_executor = create_research_agent()
    # 💡 保持 thread_id 稳定，实现持久记忆
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = str(uuid.uuid4())

config = {"configurable": {"thread_id": st.session_state.thread_id}}

st.title("🏗️ 工业级科研管线 - 性能量化版")

# --- 1. 渲染历史区 (解决覆盖问题的关键) ---
# 无论是否输入新内容，这里都会把历史的所有 human/ai 消息画出来
state = st.session_state.agent_executor.get_state(config)
if state and "messages" in state.values:
    for msg in state.values["messages"]:
        if isinstance(msg, (HumanMessage, AIMessage)) and msg.content:
            if "📝 已制定计划" in msg.content: continue # 过滤过程消息
            with st.chat_message("user" if isinstance(msg, HumanMessage) else "assistant"):
                st.markdown(msg.content)

# --- 2. 交互执行区 ---
if user_input := st.chat_input("输入科研需求..."):
    with st.chat_message("user"): st.markdown(user_input)
    
    with st.chat_message("assistant"):
        with st.status("🚀 并发 I/O & 算力调度运行中...", expanded=True) as status:
            full_start = time.time()
            # 启动流式处理
            for event in st.session_state.agent_executor.stream(
                {"input": user_input, "messages": [HumanMessage(content=user_input)], "metrics": {}}, 
                config=config, stream_mode="values"
            ):
                if event.get("plan"):
                    status.write(f"⏳ 任务栈：{len(event['plan'])} | 处理中：`{event['plan'][0]}`")
            
            # 渲染当前轮次的性能量化
            final_state = st.session_state.agent_executor.get_state(config)
            metrics = final_state.values.get("metrics", {})
            st.divider()
            st.subheader("📊 性能量化报表")
            st.table(pd.DataFrame(list(metrics.items()), columns=["环节", "耗时(s)"]))
            st.success(f"⚡ 总耗时: {time.time()-full_start:.2f}s | 并发加速已生效")
            
            # 渲染最新的总结报告
            final_msg = next((m.content for m in reversed(final_state.values["messages"]) 
                             if isinstance(m, AIMessage) and not m.tool_calls), "")
            st.markdown("### 📝 最终研究报告")
            st.markdown(final_msg)
            status.update(label="任务达标", state="complete")