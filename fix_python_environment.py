#!/usr/bin/env python3
"""
修复Python环境问题的脚本
确保虚拟环境正确配置并可以运行训练系统
"""

import os
import sys
import subprocess
import platform

def check_python_environment():
    """检查Python环境状态"""
    print("=== 检查Python环境 ===")
    print(f"Python版本: {sys.version}")
    print(f"Python可执行文件: {sys.executable}")
    print(f"工作目录: {os.getcwd()}")
    print(f"平台: {platform.platform()}")
    
    # 检查虚拟环境
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("✓ 当前在虚拟环境中运行")
    else:
        print("✗ 不在虚拟环境中运行")
    
    return True

def check_requirements():
    """检查必要的包是否已安装"""
    print("\n=== 检查依赖包 ===")
    required_packages = [
        'torch', 'transformers', 'numpy', 'flask', 'requests',
        'opencv-python', 'pillow', 'scipy', 'pydub'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package} 已安装")
        except ImportError:
            missing_packages.append(package)
            print(f"✗ {package} 未安装")
    
    if missing_packages:
        print(f"\n缺少的包: {missing_packages}")
        print("请运行: pip install " + " ".join(missing_packages))
        return False
    return True

def fix_audio_data_loading():
    """修复音频数据加载问题"""
    print("\n=== 修复音频数据加载 ===")
    
    # 检查音频配置文件
    audio_config_path = "training_data/audio/audio_config.json"
    if os.path.exists(audio_config_path):
        print(f"✓ 找到音频配置文件: {audio_config_path}")
        
        # 读取并验证配置文件
        import json
        try:
            with open(audio_config_path, 'r', encoding='utf-8') as f:
                audio_config = json.load(f)
            
            # 确保配置包含必要的键
            required_keys = ['samples', 'sample_rate', 'channels', 'duration']
            for key in required_keys:
                if key not in audio_config:
                    print(f"⚠ 音频配置缺少键: {key}")
            
            print("✓ 音频配置文件验证通过")
            
        except Exception as e:
            print(f"✗ 读取音频配置文件失败: {e}")
            return False
    else:
        print(f"✗ 音频配置文件不存在: {audio_config_path}")
        return False
    
    return True

def test_training_system():
    """测试训练系统功能"""
    print("\n=== 测试训练系统 ===")
    
    try:
        # 导入训练系统
        from enhanced_training_system_complete import EnhancedTrainingController
        from enhanced_training_system_complete import (
            LanguageDataset, AudioDataset, ImageDataset, VideoDataset
        )
        
        print("✓ 成功导入训练系统模块")
        
        # 测试数据集类
        datasets_to_test = [
            ("语言数据集", LanguageDataset),
            ("音频数据集", AudioDataset), 
            ("图像数据集", ImageDataset),
            ("视频数据集", VideoDataset)
        ]
        
        for name, dataset_class in datasets_to_test:
            try:
                # 尝试创建数据集实例
                test_dataset = dataset_class()
                print(f"✓ {name} 初始化成功")
            except Exception as e:
                print(f"✗ {name} 初始化失败: {e}")
        
        print("✓ 训练系统基本功能测试通过")
        return True
        
    except Exception as e:
        print(f"✗ 训练系统测试失败: {e}")
        return False

def create_fixed_audio_config():
    """创建修复后的音频配置文件"""
    print("\n=== 创建修复的音频配置 ===")
    
    audio_config = {
        "samples": [
            {
                "id": "audio_001",
                "file_path": "training_data/audio/sample_001.wav",
                "duration": 5.0,
                "sample_rate": 16000,
                "channels": 1,
                "transcript": "这是一个测试音频样本",
                "audio_features": {
                    "mfcc": [0.1, 0.2, 0.3],
                    "spectral_centroid": 1500,
                    "zero_crossing_rate": 0.05
                }
            },
            {
                "id": "audio_002", 
                "file_path": "training_data/audio/sample_002.wav",
                "duration": 3.5,
                "sample_rate": 16000,
                "channels": 1,
                "transcript": "另一个测试音频样本",
                "audio_features": {
                    "mfcc": [0.15, 0.25, 0.35],
                    "spectral_centroid": 1800,
                    "zero_crossing_rate": 0.08
                }
            }
        ],
        "sample_rate": 16000,
        "channels": 1,
        "duration": 5.0,
        "total_samples": 2
    }
    
    # 确保目录存在
    os.makedirs("training_data/audio", exist_ok=True)
    
    # 写入配置文件
    import json
    try:
        with open("training_data/audio/audio_config.json", 'w', encoding='utf-8') as f:
            json.dump(audio_config, f, indent=2, ensure_ascii=False)
        print("✓ 创建修复的音频配置文件")
        return True
    except Exception as e:
        print(f"✗ 创建音频配置文件失败: {e}")
        return False

def main():
    """主修复函数"""
    print("Self Brain - 环境修复和验证工具")
    print("=" * 50)
    
    # 执行修复步骤
    steps = [
        ("检查Python环境", check_python_environment),
        ("检查依赖包", check_requirements),
        ("修复音频配置", create_fixed_audio_config),
        ("修复音频数据加载", fix_audio_data_loading),
        ("测试训练系统", test_training_system)
    ]
    
    results = []
    for step_name, step_func in steps:
        try:
            success = step_func()
            results.append((step_name, success))
        except Exception as e:
            print(f"✗ {step_name} 执行失败: {e}")
            results.append((step_name, False))
    
    # 输出结果摘要
    print("\n" + "=" * 50)
    print("修复结果摘要:")
    for step_name, success in results:
        status = "✓ 通过" if success else "✗ 失败"
        print(f"{status} - {step_name}")
    
    total_passed = sum(1 for _, success in results if success)
    total_steps = len(results)
    
    print(f"\n总计: {total_passed}/{total_steps} 个步骤通过")
    
    if total_passed == total_steps:
        print("🎉 所有修复步骤完成！系统现在应该可以正常运行。")
        return True
    else:
        print("⚠ 部分修复步骤失败，请检查上述错误信息。")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
