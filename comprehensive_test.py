import requests
import json
import time

# 测试健康检查端点
def test_health_check():
    url = "http://localhost:5001/api/health"
    try:
        response = requests.get(url)
        print(f"\n健康检查端点响应状态码: {response.status_code}")
        print("响应内容:")
        print(json.dumps(response.json(), ensure_ascii=False, indent=2))
        return response.status_code == 200
    except Exception as e:
        print(f"健康检查测试失败: {e}")
        return False

# 测试获取当前情感状态
def test_get_current_emotion():
    url = "http://localhost:5001/api/emotion/current"
    try:
        response = requests.get(url)
        print(f"\n获取当前情感状态响应状态码: {response.status_code}")
        print("响应内容:")
        print(json.dumps(response.json(), ensure_ascii=False, indent=2))
        return response.status_code == 200
    except Exception as e:
        print(f"获取当前情感状态测试失败: {e}")
        return False

# 测试更新情感状态
def test_update_emotion():
    url = "http://localhost:5001/api/emotion"
    data = {"emotion": "happy"}
    try:
        response = requests.post(url, json=data)
        print(f"\n更新情感状态响应状态码: {response.status_code}")
        print("响应内容:")
        print(json.dumps(response.json(), ensure_ascii=False, indent=2))
        return response.status_code == 200
    except Exception as e:
        print(f"更新情感状态测试失败: {e}")
        return False

# 测试情感分析
def test_emotion_analysis():
    url = "http://localhost:5001/api/emotion/analyze"
    data = {"text": "I am very excited about this new project!"}
    try:
        response = requests.post(url, json=data)
        print(f"\n情感分析响应状态码: {response.status_code}")
        print("响应内容:")
        print(json.dumps(response.json(), ensure_ascii=False, indent=2))
        return response.status_code == 200
    except Exception as e:
        print(f"情感分析测试失败: {e}")
        return False

# 测试获取模型列表
def test_get_models():
    url = "http://localhost:5001/api/models"
    try:
        response = requests.get(url)
        print(f"\n获取模型列表响应状态码: {response.status_code}")
        print("响应内容:")
        print(json.dumps(response.json(), ensure_ascii=False, indent=2))
        return response.status_code == 200
    except Exception as e:
        print(f"获取模型列表测试失败: {e}")
        return False

# 测试与管理模型对话
def test_chat_with_management():
    url = "http://localhost:5001/api/chat_with_management"
    data = {"message": "Hello, how is the system working?"}
    try:
        response = requests.post(url, json=data)
        print(f"\n与管理模型对话响应状态码: {response.status_code}")
        print("响应内容:")
        print(json.dumps(response.json(), ensure_ascii=False, indent=2))
        return response.status_code == 200
    except Exception as e:
        print(f"与管理模型对话测试失败: {e}")
        return False

# 开始训练测试
def test_start_training():
    url = "http://localhost:5001/api/training/start"
    data = {"config": {"epochs": 10, "batch_size": 32}}
    try:
        response = requests.post(url, json=data)
        print(f"\n开始训练响应状态码: {response.status_code}")
        print("响应内容:")
        print(json.dumps(response.json(), ensure_ascii=False, indent=2))
        return response.status_code == 200
    except Exception as e:
        print(f"开始训练测试失败: {e}")
        return False

# 获取训练进度
def test_training_progress():
    url = "http://localhost:5001/api/training/progress"
    try:
        response = requests.get(url)
        print(f"\n获取训练进度响应状态码: {response.status_code}")
        print("响应内容:")
        print(json.dumps(response.json(), ensure_ascii=False, indent=2))
        return response.status_code == 200
    except Exception as e:
        print(f"获取训练进度测试失败: {e}")
        return False

# 停止训练测试
def test_stop_training():
    url = "http://localhost:5001/api/training/stop"
    try:
        response = requests.post(url)
        print(f"\n停止训练响应状态码: {response.status_code}")
        print("响应内容:")
        print(json.dumps(response.json(), ensure_ascii=False, indent=2))
        return response.status_code == 200
    except Exception as e:
        print(f"停止训练测试失败: {e}")
        return False

if __name__ == "__main__":
    print("开始全面测试API服务...")
    
    # 运行所有测试
    tests = [
        test_health_check,
        test_get_current_emotion,
        test_update_emotion,
        test_emotion_analysis,
        test_get_models,
        test_chat_with_management,
        test_start_training,
        test_training_progress,
        test_stop_training
    ]
    
    all_passed = True
    
    for i, test_func in enumerate(tests):
        print(f"\n=== 测试 {i+1}/{len(tests)}: {test_func.__name__} ===")
        if not test_func():
            all_passed = False
        # 短暂延迟以避免请求过于频繁
        time.sleep(1)
    
    if all_passed:
        print("\n所有测试通过！API服务功能正常。")
    else:
        print("\n部分测试失败，请检查API服务功能。")