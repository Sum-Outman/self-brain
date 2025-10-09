import sys
import json
import time
import requests
import os
from unified_language_model import UnifiedLanguageModel

# 设置测试环境  # Set up test environment
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 测试函数  # Test function
def test_unified_language_model():
    print("="*50)
    print("开始测试增强的统一语言模型训练功能")
    print("="*50)
    
    # 初始化模型  # Initialize model
    print("1. 初始化UnifiedLanguageModel...")
    model = UnifiedLanguageModel(mode="enhanced")
    print(f"模型初始化成功: {model.get_status()}")
    
    # 测试基本训练功能  # Test basic training functionality
    print("\n2. 测试基本训练功能...")
    try:
        training_result = model.train_model(
            data_path="./data",
            languages=["zh"],
            epochs=2,
            batch_size=4,
            learning_rate=0.0001
        )
        print(f"基本训练成功！训练结果: {training_result}")
        print(f"训练历史: {model.get_training_history()}")
    except Exception as e:
        print(f"基本训练失败: {str(e)}")
    
    # 测试增量学习功能  # Test incremental learning functionality
    print("\n3. 测试增量学习功能...")
    try:
        incremental_result = model.incremental_train(
            data_path="./data",
            languages=["en"],
            epochs=2,
            batch_size=4,
            learning_rate=0.00001
        )
        print(f"增量学习成功！结果: {incremental_result}")
        print(f"低置信度样本数量: {len(model.get_low_confidence_samples())}")
    except Exception as e:
        print(f"增量学习失败: {str(e)}")
    
    # 测试知识迁移功能  # Test knowledge transfer functionality
    print("\n4. 测试知识迁移功能...")
    try:
        transfer_result = model.transfer_learn(
            source_language="en",
            target_language=["de"],
            data_path="./data",
            epochs=2,
            batch_size=4,
            learning_rate=0.00001
        )
        print(f"知识迁移成功！结果: {transfer_result}")
    except Exception as e:
        print(f"知识迁移失败: {str(e)}")
    
    # 测试模型评估  # Test model evaluation
    print("\n5. 测试模型评估功能...")
    try:
        evaluation_result = model.evaluate_model(
            data_path="./data",
            languages=["zh", "en", "de"]
        )
        print(f"模型评估成功！评估结果: {evaluation_result}")
    except Exception as e:
        print(f"模型评估失败: {str(e)}")
    
    # 测试保存和加载模型  # Test save and load model
    print("\n6. 测试保存和加载模型...")
    try:
        model_path = f"./models/test_model_{int(time.time())}"
        model.save_model(model_path)
        print(f"模型保存成功: {model_path}")
        
        new_model = UnifiedLanguageModel(mode="enhanced")
        new_model.load_model(model_path)
        print(f"模型加载成功！状态: {new_model.get_status()}")
    except Exception as e:
        print(f"模型保存/加载失败: {str(e)}")
    
    print("\n" + "="*50)
    print("增强的统一语言模型训练功能测试完成")
    print("="*50)

# 测试API接口  # Test API interface
def test_api_endpoints():
    print("\n" + "="*50)
    print("开始测试API接口功能")
    print("="*50)
    
    base_url = "http://localhost:5002"
    
    # 测试健康检查接口  # Test health check interface
    print("1. 测试健康检查接口...")
    try:
        response = requests.get(f"{base_url}/health", timeout=2)
        print(f"健康检查结果: {response.json()}")
    except Exception as e:
        print(f"健康检查失败: {str(e)} | 请确保服务已启动")
    
    # 测试训练接口  # Test training interface
    print("\n2. 测试训练接口...")
    try:
        training_config = {
            "model_id": "B_language_test",
            "epochs": 2,
            "batch_size": 4,
            "learning_rate": 0.0001,
            "languages": ["zh", "en"]
        }
        response = requests.post(f"{base_url}/train", json=training_config, timeout=30)
        print(f"训练接口响应: {response.json()}")
    except Exception as e:
        print(f"训练接口测试失败: {str(e)} | 请确保服务已启动")
    
    # 测试增量学习API  # Test incremental learning API
    print("\n3. 测试增量学习API...")
    try:
        incremental_config = {
            "model_id": "B_language_test",
            "epochs": 2,
            "batch_size": 4,
            "learning_rate": 0.00001,
            "languages": ["de"],
            "use_incremental": True
        }
        response = requests.post(f"{base_url}/train", json=incremental_config, timeout=30)
        print(f"增量学习接口响应: {response.json()}")
    except Exception as e:
        print(f"增量学习接口测试失败: {str(e)} | 请确保服务已启动")
    
    print("\n" + "="*50)
    print("API接口功能测试完成")
    print("="*50)

if __name__ == "__main__":
    # 创建模型目录（如果不存在）  # Create model directory if not exists
    if not os.path.exists("./models"):
        os.makedirs("./models")
    
    # 测试直接模型功能  # Test direct model functionality
    test_unified_language_model()
    
    # 询问用户是否要测试API接口  # Ask user if they want to test API endpoints
    test_api = input("\n是否要测试API接口？(y/n): ")
    if test_api.lower() == 'y':
        test_api_endpoints()
    else:
        print("跳过API接口测试。")
        print("\n测试完成！请确保B_language服务已启动，然后可以通过API进行进一步测试。")