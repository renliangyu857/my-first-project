import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# 直接测试底层连接
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_API_BASE")
)

print(f"📡 正在测试地址: {os.getenv('OPENAI_API_BASE')}")

try:
    # 1. 先测试最简单的文本对话（不带工具）
    print("🧪 测试 1：简单对话...")
    response = client.chat.completions.create(
        model="gpt-5.2", # 如果报错 404，尝试换成 gpt-3.5-turbo
        messages=[{"role": "user", "content": "你好，请回复'连接成功'"}]
    )
    print(f"✅ 测试 1 成功！回复内容: {response.choices[0].message.content}")

    # 2. 测试并发工具格式（这是最容易报 model_dump 错的地方）
    print("\n🧪 测试 2：工具定义测试...")
    response = client.chat.completions.create(
        model="- gpt-5.2",
        messages=[{"role": "user", "content": "给我 2 个苹果和 3 个香蕉"}],
        tools=[{
            "type": "function",
            "function": {
                "name": "get_fruit",
                "parameters": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}, "count": {"type": "integer"}}
                }
            }
        }]
    )
    print(f"✅ 测试 2 成功！模型生成的工具指令: {response.choices[0].message.tool_calls}")

except Exception as e:
    print(f"\n❌ 捕获到错误！")
    print(f"错误类型: {type(e).__name__}")
    print(f"错误详情: {e}")
    if "404" in str(e):
        print("💡 建议：尝试把 OPENAI_API_BASE 末尾的 /v1 去掉试试，或者检查模型名称是否写错。")