import requests
import json

# 测试通过web_interface与A Management Model对话
def test_web_interface_chat():
    print("=== 测试通过web_interface与A Management Model对话 ===")
    try:
        url = "http://localhost:5000/api/chat/send"
        message = "Hello, this is a test message through web interface"
        
        response = requests.post(
            url,
            json={
                "message": message,
                "conversation_id": "test_conv",
                "knowledge_base": "all"
            },
            headers={
                'Content-Type': 'application/json'
            },
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"响应状态: {result.get('status')}")
            print(f"响应内容: {result.get('response')}")
        else:
            print(f"请求失败: HTTP {response.status_code}")
            print(f"错误内容: {response.text}")
    except Exception as e:
        print(f"测试失败: {str(e)}")

# 直接测试与A Management Model的对话
def test_direct_manager_chat():
    print("\n=== 直接测试与A Management Model的对话 ===")
    try:
        url = "http://localhost:5015/api/chat"
        message = "List all available models"
        
        response = requests.post(
            url,
            json={
                "message": message
            },
            headers={
                'Content-Type': 'application/json'
            },
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"响应状态: {result.get('status')}")
            print(f"模型: {result.get('model')}")
            print(f"响应内容:\n{result.get('response')}")
            print(f"对话数据: {result.get('conversation_data')}")
        else:
            print(f"请求失败: HTTP {response.status_code}")
            print(f"错误内容: {response.text}")
    except Exception as e:
        print(f"测试失败: {str(e)}")

if __name__ == "__main__":
    # 运行测试
    test_web_interface_chat()
    test_direct_manager_chat()