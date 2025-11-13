import os
import requests
from openai import OpenAI

print("🔍 === 项目环境全面检查开始 ===\n")

# 1️⃣ 检查当前目录
print(f"📂 当前工作目录: {os.getcwd()}")
print()

# 2️⃣ 检查 OPENAI_API_KEY
api_key = os.getenv("OPENAI_API_KEY")
if api_key:
    print("✅ 环境变量 OPENAI_API_KEY 已加载")
else:
    print("❌ 环境变量 OPENAI_API_KEY 未设置！请先运行：")
    print('   setx OPENAI_API_KEY "你的API密钥"')
    print("   （然后关闭并重新打开 PowerShell 再运行）")
    exit(1)
print()

# 3️⃣ 检查 GPT 调用
try:
    print("🧠 测试 OpenAI API 调用中...")
    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Reply with 'Hello from GPT'"}]
    )
    print("✅ GPT 调用成功:", response.choices[0].message.content)
except Exception as e:
    print("❌ GPT 调用失败:", e)
    exit(1)
print()

# 4️⃣ 检查 FastAPI /ping
try:
    print("🌐 测试 FastAPI 后端 /ping 接口...")
    r = requests.get("http://127.0.0.1:8001/ping", timeout=5)
    if r.status_code == 200:
        print("✅ FastAPI /ping 响应成功:", r.text)
    else:
        print(f"⚠️ FastAPI 返回状态码: {r.status_code}")
except Exception as e:
    print("❌ FastAPI 未启动或 /ping 路径不存在:", e)
    print("👉 请确认是否在 backend 目录运行：")
    print("   uvicorn app:app --reload --port 8001")
print()

print("🎯 检查结束！如果上面三项都是 ✅，环境一切正常。")
