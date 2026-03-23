import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool

# 1. 加载配置
load_dotenv()

# 2. 定义两个极其简单的“空工具”用于测试并发
@tool
def tool_apple(n: int):
    """当提到苹果时调用"""
    return f"获取了 {n} 个苹果"

@tool
def tool_banana(n: int):
    """当提到香蕉时调用"""
    return f"获取了 {n} 个香蕉"

def verify_connection():
    print("🚀 开始验证 Codex 中转站连接...")
    
    # 3. 初始化 LLM
    # 请确保你的 .env 中有 OPENAI_API_KEY 和 OPENAI_API_BASE
    llm = ChatOpenAI(
        model="gpt-4o",  # 或者换成你卡片支持的其他模型名
        temperature=0,
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_API_BASE")
    )

    # 绑定工具，开启并发支持
    tools = [tool_apple, tool_banana]
    llm_with_tools = llm.bind_tools(tools, parallel_tool_calls=True)

    # 4. 发起一个明确要求并发的测试请求
    print("📡 正在发送并发工具调用请求：'给我 2 个苹果和 3 个香蕉'...")
    try:
        res = llm_with_tools.invoke("给我 2 个苹果和 3 个香蕉")
        
        # 5. 结果分析
        tool_calls = res.tool_calls
        print(f"\n✅ 响应成功！")
        print(f"🤖 模型回复内容: {res.content}")
        print(f"🛠️ 工具调用数量: {len(tool_calls)}")
        
        for i, call in enumerate(tool_calls):
            print(f"   - 调用 {i+1}: {call['name']}({call['args']})")

        if len(tool_calls) > 1:
            print("\n🌟 恭喜！你的中转站完美支持【并发工具调用】，LangGraph 将会飞速运行。")
        else:
            print("\n⚠️ 模型虽然响应了，但没有同时发起两个调用。可能是模型版本或中转站限制。")

    except Exception as e:
        print(f"\n❌ 调用失败！错误详情: {e}")
        print("💡 请检查：1. API Key 是否正确；2. Base URL 是否包含 /v1；3. 余额是否充足。")

if __name__ == "__main__":
    verify_connection()