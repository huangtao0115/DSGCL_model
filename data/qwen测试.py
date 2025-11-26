import openai
import requests

# 配置参数 - 请务必替换为你的实际信息
QWEN_API_BASE = "https://dashscope.aliyuncs.com/api/v1"  # 注意保留/api/v1
QWEN_API_KEY = " "  # 替换为你的API密钥
MODEL_NAME = "qwen-vl-max"  # 确认模型名是否完全一致

# 设置OpenAI客户端（兼容模式）
openai.api_base = QWEN_API_BASE
openai.api_key = QWEN_API_KEY

# 创建一个非常简单的请求试试水
try:
    # 尝试列出模型，有时这个端点能帮你确认认证和基础URL是否正确
    # 注意：并非所有提供商都支持此端点，但值得一试
    test_url = f"{QWEN_API_BASE}/models"
    headers = {
        "Authorization": f"Bearer {QWEN_API_KEY}",
        "Content-Type": "application/json"
    }
    response = requests.get(test_url, headers=headers, timeout=10)
    print(f"测试列表模型端点状态码: {response.status_code}")
    if response.status_code == 200:
        print("✅ 模型列表请求成功，至少认证和基础URL可能没问题")
        # 打印返回的模型列表看看是否包含你想要的模型
        print("返回信息:", response.text)
    else:
        print(f"❌ 模型列表请求失败: {response.status_code} - {response.text}")

except Exception as e:
    print(f"测试列表模型请求时发生异常: {e}")

# 尝试一个最简单的对话请求
try:
    print("\n尝试发送一个最简单的对话请求...")
    response = openai.ChatCompletion.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": "请说'Hello, World!'"}],
        max_tokens=10,
        timeout=15
    )
    print("✅ 简单对话请求成功！")
    print("回复:", response['choices'][0]['message']['content'])

except openai.error.InvalidRequestError as e:
    print(f"❌ 无效请求（可能是模型名称、参数或权限问题）: {e}")
except openai.error.AuthenticationError as e:
    print(f"❌ 认证失败（API密钥或权限问题）: {e}")
except openai.error.APIError as e:
    print(f"❌ API错误（HTTP {e.http_status}）: {e}")
except Exception as e:
    print(f"❌ 其他错误: {e}")