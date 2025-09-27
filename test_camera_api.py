import requests
import json
import time

# 测试基础URL
base_url = 'http://localhost:5000/api/camera'

print("=== 测试相机API ===")

# 1. 测试获取可用相机列表
try:
    print("\n1. 获取可用相机列表...")
    response = requests.get(f'{base_url}/inputs')
    print(f"状态码: {response.status_code}")
    cameras = response.json()
    print(f"可用相机数量: {len(cameras)}")
    for camera in cameras:
        print(f"  - 相机ID: {camera['id']}, 名称: {camera['name']}")
except Exception as e:
    print(f"获取相机列表失败: {str(e)}")

# 2. 尝试启动相机 0
camera_id = 0
try:
    print(f"\n2. 启动相机 {camera_id}...")
    response = requests.post(f'{base_url}/start/{camera_id}', json={"resolution": "1280x720"})
    print(f"状态码: {response.status_code}")
    result = response.json()
    print(f"结果: {result}")
except Exception as e:
    print(f"启动相机 {camera_id} 失败: {str(e)}")

# 3. 获取相机状态
try:
    print(f"\n3. 获取相机 {camera_id} 状态...")
    response = requests.get(f'{base_url}/settings/{camera_id}')
    print(f"状态码: {response.status_code}")
    status = response.json()
    print(f"相机设置: {status}")
except Exception as e:
    print(f"获取相机 {camera_id} 状态失败: {str(e)}")

# 4. 获取相机帧
try:
    print(f"\n4. 获取相机 {camera_id} 帧...")
    response = requests.get(f'{base_url}/frame/{camera_id}')
    print(f"状态码: {response.status_code}")
    frame_data = response.json()
    print(f"帧数据: {'成功获取' if frame_data.get('status') == 'success' else frame_data}")
except Exception as e:
    print(f"获取相机 {camera_id} 帧失败: {str(e)}")

# 5. 拍摄快照
try:
    print(f"\n5. 拍摄相机 {camera_id} 快照...")
    response = requests.post(f'{base_url}/take-snapshot/{camera_id}')
    print(f"状态码: {response.status_code}")
    snapshot = response.json()
    print(f"快照结果: {'成功获取' if snapshot.get('status') == 'success' else snapshot}")
except Exception as e:
    print(f"拍摄相机 {camera_id} 快照失败: {str(e)}")

# 6. 停止相机
try:
    print(f"\n6. 停止相机 {camera_id}...")
    response = requests.post(f'{base_url}/stop/{camera_id}')
    print(f"状态码: {response.status_code}")
    stop_result = response.json()
    print(f"停止结果: {stop_result}")
except Exception as e:
    print(f"停止相机 {camera_id} 失败: {str(e)}")

print("\n=== 测试完成 ===")