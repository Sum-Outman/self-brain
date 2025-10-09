import requests
import json

# 测试启动相机API
def test_start_camera():
    try:
        # 相机ID为0
        camera_id = 0
        
        # 发送POST请求
        url = f"http://localhost:5000/api/camera/start/{camera_id}"
        headers = {'Content-Type': 'application/json'}
        data = {}
        
        response = requests.post(url, headers=headers, data=json.dumps(data))
        
        print(f"测试启动相机 {camera_id}:")
        print(f"状态码: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"响应内容: {json.dumps(result, indent=2)}")
            print("\n✅ 相机启动成功！")
        else:
            print(f"❌ 相机启动失败: {response.text}")
            
    except Exception as e:
        print(f"❌ 发生错误: {str(e)}")

if __name__ == "__main__":
    print("=== 相机启动API测试 ===")
    test_start_camera()
    print("=== 测试完成 ===")