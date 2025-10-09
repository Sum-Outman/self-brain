import requests
import time
import json

# 配置服务器地址
BASE_URL = "http://localhost:5000"

print("Testing self-learning functionality...")

# 1. 获取当前的自主学习参数
print("\n1. Getting current self-learning parameters:")
try:
    response = requests.get(f"{BASE_URL}/self_learning/params")
    if response.status_code == 200:
        params = response.json()
        print(f"Current parameters: {json.dumps(params, indent=2)}")
    else:
        print(f"Failed to get parameters: {response.status_code} - {response.text}")
except Exception as e:
    print(f"Error getting parameters: {e}")

# 2. 更新自主学习参数
print("\n2. Updating self-learning parameters:")
new_params = {
    "enable_self_learning": True,
    "learning_rate": 0.002,
    "confidence_threshold": 0.75,
    "exploration_factor": 0.25,
    "gap_detection_interval": 120,
    "external_source_interval": 600
}

try:
    response = requests.post(
        f"{BASE_URL}/self_learning/params",
        headers={"Content-Type": "application/json"},
        data=json.dumps(new_params)
    )
    if response.status_code == 200:
        result = response.json()
        print(f"Update result: {json.dumps(result, indent=2)}")
    else:
        print(f"Failed to update parameters: {response.status_code} - {response.text}")
except Exception as e:
    print(f"Error updating parameters: {e}")

# 3. 验证参数是否已更新
print("\n3. Verifying parameters were updated:")
try:
    response = requests.get(f"{BASE_URL}/self_learning/params")
    if response.status_code == 200:
        updated_params = response.json()
        print(f"Updated parameters: {json.dumps(updated_params, indent=2)}")
        
        # 验证关键参数是否匹配
        for key, value in new_params.items():
            if key in updated_params and updated_params[key] == value:
                print(f"✓ {key} correctly updated to {value}")
            else:
                print(f"✗ {key} not updated correctly. Expected: {value}, Got: {updated_params.get(key)}")
    else:
        print(f"Failed to get updated parameters: {response.status_code} - {response.text}")
except Exception as e:
    print(f"Error getting updated parameters: {e}")

# 4. 测试知识查询以触发自主学习
print("\n4. Testing knowledge query to trigger self-learning:")
test_query = {
    "query": "What is artificial intelligence?",
    "user_id": "test_user"
}

try:
    response = requests.post(
        f"{BASE_URL}/query",
        headers={"Content-Type": "application/json"},
        data=json.dumps(test_query)
    )
    if response.status_code == 200:
        result = response.json()
        print(f"Query result received: {result.get('response')[:100]}...")
    else:
        print(f"Query failed: {response.status_code} - {response.text}")
except Exception as e:
    print(f"Error making query: {e}")

print("\nSelf-learning functionality test completed.")