import requests
import json

# 测试健康检查端点
def test_health_check():
    url = "http://localhost:5001/api/health"
    try:
        response = requests.get(url)
        print(f"健康检查端点响应状态码: {response.status_code}")
        print("响应内容:")
        print(json.dumps(response.json(), ensure_ascii=False, indent=2))
        return response.status_code == 200
    except Exception as e:
        print(f"测试失败: {e}")
        return False

if __name__ == "__main__":
    print("开始测试API服务...")
    if test_health_check():
        print("API服务测试成功!")
    else:
        print("API服务测试失败!")