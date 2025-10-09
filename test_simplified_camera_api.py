import sys
import os
from datetime import datetime

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入CameraManager
try:
    from camera_manager import get_camera_manager
    print("成功导入CameraManager")
    camera_manager = get_camera_manager()
except Exception as e:
    print(f"导入CameraManager失败: {str(e)}")
    sys.exit(1)

print("=== 简化版相机API测试 ===")

# 1. 测试列出可用相机
try:
    print("\n1. 列出可用相机...")
    cameras = camera_manager.list_available_cameras()
    print(f"可用相机数量: {len(cameras)}")
    for camera in cameras:
        print(f"  - 相机ID: {camera['id']}, 名称: {camera['name']}, 分辨率: {camera['width']}x{camera['height']}")
except Exception as e:
    print(f"列出相机失败: {str(e)}")

# 2. 尝试启动相机0
try:
    camera_id = 0
    print(f"\n2. 启动相机 {camera_id}...")
    # 准备相机参数，模拟API调用中的分辨率参数
    params = {"width": 1280, "height": 720}
    success = camera_manager.start_camera(camera_id, params)
    if success:
        print(f"成功启动相机 {camera_id}")
        # 获取相机状态
        status = camera_manager.get_camera_status(camera_id)
        print(f"相机状态: {status}")
        
        # 获取相机设置
        settings = camera_manager.get_camera_settings(camera_id)
        print(f"相机设置: {settings}")
    else:
        print(f"启动相机 {camera_id} 失败")
except Exception as e:
    print(f"启动相机失败: {str(e)}")

# 3. 测试获取相机帧
try:
    if camera_manager.get_camera_status(camera_id).get("is_active"):
        print(f"\n3. 获取相机 {camera_id} 帧...")
        frame_data = camera_manager.get_camera_frame(camera_id)
        if frame_data:
            print(f"成功获取帧数据，大小: {len(frame_data.get('frame', ''))} 字节")
            print(f"时间戳: {frame_data.get('timestamp')}")
        else:
            print("获取帧数据失败")
except Exception as e:
    print(f"获取帧失败: {str(e)}")

# 4. 测试拍摄快照
try:
    if camera_manager.get_camera_status(camera_id).get("is_active"):
        print(f"\n4. 拍摄相机 {camera_id} 快照...")
        snapshot = camera_manager.take_snapshot(camera_id)
        if snapshot:
            print(f"成功拍摄快照: {snapshot.get('snapshot_id')}")
            print(f"时间戳: {snapshot.get('timestamp')}")
        else:
            print("拍摄快照失败")
except Exception as e:
    print(f"拍摄快照失败: {str(e)}")

# 5. 尝试停止相机
try:
    if camera_manager.get_camera_status(camera_id).get("is_active"):
        print(f"\n5. 停止相机 {camera_id}...")
        success = camera_manager.stop_camera(camera_id)
        if success:
            print(f"成功停止相机 {camera_id}")
        else:
            print(f"停止相机 {camera_id} 失败")
except Exception as e:
    print(f"停止相机失败: {str(e)}")

# 6. 测试我们添加的API函数的逻辑
try:
    print("\n6. 测试根级相机API逻辑...")
    # 模拟GET请求逻辑
    available_cameras = camera_manager.list_available_cameras()
    active_camera_ids = camera_manager.get_active_camera_ids()
    
    # 构建与我们添加的API函数返回类似的响应
    mock_response = {
        "status": "success",
        "available_cameras": available_cameras,
        "active_camera_count": len(active_camera_ids),
        "active_camera_ids": active_camera_ids,
        "api_version": "1.0"
    }
    
    print("GET请求模拟响应:")
    print(f"  可用相机数量: {mock_response['available_cameras']}")
    print(f"  活动相机数量: {mock_response['active_camera_count']}")
    print(f"  活动相机ID: {mock_response['active_camera_ids']}")
    
    # 模拟POST请求逻辑 - 启动相机
    print("\nPOST请求模拟 (启动相机):")
    mock_post_data = {
        "camera_id": 1,
        "operation": "start",
        "resolution": "640x480"
    }
    
    width, height = map(int, mock_post_data['resolution'].split('x'))
    params = {"width": width, "height": height}
    success = camera_manager.start_camera(mock_post_data['camera_id'], params)
    
    if success:
        print(f"  成功启动相机 {mock_post_data['camera_id']}")
    else:
        print(f"  启动相机 {mock_post_data['camera_id']} 失败")
        
    # 停止刚刚启动的相机
    if success:
        camera_manager.stop_camera(mock_post_data['camera_id'])
except Exception as e:
    print(f"测试API逻辑失败: {str(e)}")

print("\n=== 简化版测试完成 ===")