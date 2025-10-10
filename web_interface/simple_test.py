import os
import sys
import requests
import json
import time

def test_system_status():
    """测试系统状态API"""
    print("=== 测试系统状态 ===")
    try:
        response = requests.get('http://localhost:5015/api/system_status')
        if response.status_code == 200:
            data = response.json()
            print(f"系统状态: {data.get('status')}")
            print(f"模型总数: {data.get('models', {}).get('total')}")
            print(f"活跃模型数: {data.get('models', {}).get('active')}")
            return True
        else:
            print(f"系统状态API请求失败: {response.status_code}")
            return False
    except Exception as e:
        print(f"系统状态测试异常: {str(e)}")
        return False

def test_model_list():
    """测试模型列表API"""
    print("\n=== 测试模型列表 ===")
    try:
        response = requests.get('http://localhost:5015/api/models')
        if response.status_code == 200:
            data = response.json()
            if data.get('status') == 'success':
                models = data.get('models', [])
                print(f"发现 {len(models)} 个模型:")
                for model in models:
                    print(f"- {model.get('model_id')}: {model.get('name')} (状态: {model.get('status')})")
                return True
            else:
                print(f"获取模型列表失败: {data.get('message')}")
                return False
        else:
            print(f"模型列表API请求失败: {response.status_code}")
            return False
    except Exception as e:
        print(f"模型列表测试异常: {str(e)}")
        return False

def test_training_status():
    """测试训练状态API"""
    print("\n=== 测试训练状态 ===")
    try:
        response = requests.get('http://localhost:5015/api/training/status')
        if response.status_code == 200:
            data = response.json()
            if data.get('status') == 'success':
                training_data = data.get('training', {})
                print(f"活跃训练会话数: {training_data.get('active_sessions')}")
                print(f"总训练会话数: {training_data.get('total_sessions')}")
                print(f"完成训练会话数: {training_data.get('completed_sessions')}")
                print(f"失败训练会话数: {training_data.get('failed_sessions')}")
                
                # 检查系统资源
                system_data = data.get('system', {})
                print(f"CPU使用率: {system_data.get('cpu_usage')}%")
                print(f"内存使用率: {system_data.get('memory_usage', {}).get('percent')}%")
                print(f"GPU数量: {len(system_data.get('gpu_info', []))}")
                
                return True
            else:
                print(f"获取训练状态失败")
                return False
        else:
            print(f"训练状态API请求失败: {response.status_code}")
            return False
    except Exception as e:
        print(f"训练状态测试异常: {str(e)}")
        return False

def test_knowledge_base():
    """测试知识库功能"""
    print("\n=== 测试知识库功能 ===")
    try:
        # 检查知识库存储目录
        kb_dir = os.path.join('knowledge_base_storage')
        if os.path.exists(kb_dir):
            items = os.listdir(kb_dir)
            print(f"知识库存储目录存在，包含 {len(items)} 个项目")
        else:
            print(f"知识库存储目录不存在: {kb_dir}")
            
        # 尝试获取知识库优化状态
        response = requests.post('http://localhost:5015/api/knowledge/optimize')
        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                print(f"知识库优化结果: {data.get('message')}")
            else:
                print(f"知识库优化失败: {data.get('message')}")
        else:
            print(f"知识库优化API请求失败: {response.status_code}")
        
        return True
    except Exception as e:
        print(f"知识库测试异常: {str(e)}")
        return False

def test_model_configs():
    """测试模型配置文件"""
    print("\n=== 测试模型配置 ===")
    models_dir = os.path.join('models')
    if not os.path.exists(models_dir):
        print(f"模型目录不存在: {models_dir}")
        return False
    
    model_folders = os.listdir(models_dir)
    print(f"发现 {len(model_folders)} 个模型配置目录:")
    
    for folder in model_folders:
        config_path = os.path.join(models_dir, folder, 'config.json')
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                print(f"- {folder}: 配置文件存在 (provider: {config.get('provider')}, source: {config.get('model_source')})")
            except Exception as e:
                print(f"- {folder}: 配置文件读取失败: {str(e)}")
        else:
            print(f"- {folder}: 配置文件不存在")
    
    return True

def main():
    """主测试函数"""
    print("开始Self Brain系统测试...")
    
    # 记录开始时间
    start_time = time.time()
    
    # 运行各项测试
    tests = [
        test_system_status,
        test_model_list,
        test_training_status,
        test_knowledge_base,
        test_model_configs
    ]
    
    # 执行测试并收集结果
    results = []
    for test in tests:
        result = test()
        results.append((test.__name__, result))
    
    # 计算测试时间
    elapsed_time = time.time() - start_time
    
    # 输出测试结果摘要
    print("\n=== 测试结果摘要 ===")
    success_count = sum(1 for _, success in results if success)
    
    for test_name, success in results:
        status = "通过" if success else "失败"
        print(f"- {test_name}: {status}")
    
    print(f"\n测试完成，共 {len(results)} 项测试，{success_count} 项通过，耗时 {elapsed_time:.2f} 秒")
    
    # 如果有测试失败，返回错误代码
    if success_count < len(results):
        print("\n系统存在问题，需要修复")
        sys.exit(1)
    else:
        print("\n系统基本功能测试通过")
        sys.exit(0)

if __name__ == "__main__":
    main()