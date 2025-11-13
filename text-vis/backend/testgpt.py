from openai import OpenAI
import os

# ✅ 从环境变量读取密钥
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    print("❌ 没检测到 OPENAI_API_KEY，请先运行 setx OPENAI_API_KEY \"sk-xxxxx\" 并重启 PowerShell。")
    exit()

client = OpenAI(api_key=api_key)

print("🔹 正在调用 GPT 模型测试中...")

try:
    resp = client.responses.create(
        model="gpt-3.5-turbo",
        input="Say hello in one short sentence."
    )
    print("✅ 调用成功！GPT 返回内容：")
    print(resp.output[0].content[0].text)
except Exception as e:
    print("❌ 调用失败：")
    print(e)
