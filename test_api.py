import requests
import time

def test_api_endpoint(url, description):
    """测试API端点并显示结果"""
    print(f"\n=== 测试 {description} ({url}) ===")
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            print(f"✅ 成功连接到 {description}")
            try:
                data = response.json()
                print(f"   响应状态: {data.get('status', 'unknown')}")
                if 'models' in data:
                    print(f"   模型数量: {len(data['models'])}")
            except:
                print(f"   响应内容: {response.text[:100]}...")
        else:
            print(f"❌ 连接失败，状态码: {response.status_code}")
    except Exception as e:
        print(f"❌ 连接异常: {str(e)}")

if __name__ == "__main__":
    # 测试Web界面API
    test_api_endpoint("http://localhost:5000/api/models", "Web界面模型列表")
    test_api_endpoint("http://localhost:5000/api/system/status", "Web界面系统状态")
    
    # 测试Manager Model API
    test_api_endpoint("http://localhost:5015/api/models", "Manager模型列表")
    test_api_endpoint("http://localhost:5015/api/health", "Manager健康检查")
    
    print("\n测试完成")