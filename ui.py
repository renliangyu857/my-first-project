import streamlit as st
import time, os, pandas as pd
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage
from agent import create_research_agent

load_dotenv()
st.set_page_config(page_title="科研助理 v2.5", layout="wide")

# 侧边栏 ID 管理
with st.sidebar:
    st.header("⚙️ 记忆管理")
    saved_id = st.text_input("会话 Thread ID", value="researcher_01")
    if st.button("找回记忆"):
        st.session_state.thread_id = saved_id
        st.rerun()

if "thread_id" not in st.session_state:
    st.session_state.thread_id = saved_id

if "agent_executor" not in st.session_state:
    try:
        st.session_state.agent_executor = create_research_agent()
    except Exception as e:
        st.error(f"❌ 引擎启动失败: {e}") # 💡 这样即使报错你也能看到原因，而不是输入框消失

config = {"configurable": {"thread_id": st.session_state.thread_id}}

st.title("🏗️ 工业级科研管线")

# 1. 先渲染历史记录
try:
    state = st.session_state.agent_executor.get_state(config)
    if state and "messages" in state.values:
        for msg in state.values["messages"]:
            if isinstance(msg, (HumanMessage, AIMessage)) and msg.content:
                with st.chat_message("user" if isinstance(msg, HumanMessage) else "assistant"):
                    st.markdown(msg.content)
except:
    st.warning("暂无历史记忆")

# 2. 交互输入框（确保它在最外层）
if user_input := st.chat_input("输入科研需求..."):
    with st.chat_message("user"): st.markdown(user_input)
    
    with st.chat_message("assistant"):
        with st.status("🚀 运行中...", expanded=True) as status:
            full_start = time.time()
            # 这里的流式输出逻辑保持不变...
            for event in st.session_state.agent_executor.stream(
                {"input": user_input, "messages": [HumanMessage(content=user_input)], "metrics": {}}, 
                config=config, stream_mode="values"
            ):
                pass 
            
            final_state = st.session_state.agent_executor.get_state(config)
            metrics = final_state.values.get("metrics", {})
            st.table(pd.DataFrame(list(metrics.items()), columns=["环节", "耗时(s)"]))
            
            final_msg = next((m.content for m in reversed(final_state.values["messages"]) 
                             if isinstance(m, AIMessage) and not m.tool_calls), "处理完成")
            st.markdown(final_msg)
            status.update(label="任务完成", state="complete")