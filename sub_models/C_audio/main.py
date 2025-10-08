# -*- coding: utf-8 -*-
# Copyright 2025 The AGI Brain System Authors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
音频处理模型主程序 | Audio Processing Model Main Program
音频处理模型的主要入口点和集成接口
Main entry point and integration interface for audio processing model
"""

import os
import sys
import logging
import argparse
from pathlib import Path

# 添加当前目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from enhanced_audio_model import EnhancedAudioProcessingModel, AudioProcessingMode, AudioEffectType
from config_loader import AudioConfigLoader

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("audio_model_main.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class AudioModelMain:
    """
    音频处理模型主类 | Audio Processing Model Main Class
    提供音频处理模型的主要功能和集成接口
    Provides main functionality and integration interface for audio processing model
    """
    
    def __init__(self, config_path: str = None):
        """
        初始化音频处理模型主程序 | Initialize audio processing model main program
        
        参数 Parameters:
        config_path: 配置文件路径 | Configuration file path
        """
        self.logger = logger
        self.config_loader = AudioConfigLoader(config_path)
        self.config = self.config_loader.load_config()
        self.audio_model = None
        self.initialize_model()
    
    def initialize_model(self):
        """初始化音频处理模型 | Initialize audio processing model"""
        try:
            self.logger.info("正在初始化音频处理模型... | Initializing audio processing model...")
            self.audio_model = EnhancedAudioProcessingModel(self.config)
            self.logger.info("音频处理模型初始化成功 | Audio processing model initialized successfully")
            return True
        except Exception as e:
            self.logger.error(f"模型初始化失败: {e} | Model initialization failed: {e}")
            return False
    
    def process_audio(self, audio_data, mode: AudioProcessingMode, **kwargs):
        """
        处理音频数据 | Process audio data
        
        参数 Parameters:
        audio_data: 音频数据 | Audio data
        mode: 处理模式 | Processing mode
        **kwargs: 额外参数 | Additional parameters
        
        返回 Returns:
        处理结果 | Processing result
        """
        try:
            self.logger.info(f"开始处理音频，模式: {mode.value} | Starting audio processing, mode: {mode.value}")
            result = self.audio_model.process_audio(audio_data, mode, **kwargs)
            self.logger.info(f"音频处理完成 | Audio processing completed")
            return result
        except Exception as e:
            self.logger.error(f"音频处理失败: {e} | Audio processing failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "mode": mode.value
            }
    
    def apply_audio_effect(self, audio_data, effect_type: AudioEffectType, **kwargs):
        """
        应用音频特效 | Apply audio effect
        
        参数 Parameters:
        audio_data: 音频数据 | Audio data
        effect_type: 特效类型 | Effect type
        **kwargs: 特效参数 | Effect parameters
        
        返回 Returns:
        特效应用结果 | Effect application result
        """
        try:
            self.logger.info(f"开始应用音频特效: {effect_type.value} | Starting audio effect application: {effect_type.value}")
            result = self.audio_model.apply_audio_effect(audio_data, effect_type, **kwargs)
            self.logger.info(f"音频特效应用完成 | Audio effect application completed")
            return result
        except Exception as e:
            self.logger.error(f"音频特效应用失败: {e} | Audio effect application failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "effect_type": effect_type.value
            }
    
    def start_realtime_processing(self, callback=None):
        """
        启动实时处理 | Start real-time processing
        
        参数 Parameters:
        callback: 回调函数 | Callback function
        
        返回 Returns:
        启动结果 | Start result
        """
        try:
            self.logger.info("启动实时音频处理 | Starting real-time audio processing")
            result = self.audio_model.start_realtime_processing(callback)
            self.logger.info("实时音频处理已启动 | Real-time audio processing started")
            return result
        except Exception as e:
            self.logger.error(f"启动实时处理失败: {e} | Failed to start real-time processing: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def stop_realtime_processing(self):
        """停止实时处理 | Stop real-time processing"""
        try:
            self.logger.info("停止实时音频处理 | Stopping real-time audio processing")
            result = self.audio_model.stop_realtime_processing()
            self.logger.info("实时音频处理已停止 | Real-time audio processing stopped")
            return result
        except Exception as e:
            self.logger.error(f"停止实时处理失败: {e} | Failed to stop real-time processing: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def get_model_status(self):
        """获取模型状态 | Get model status"""
        try:
            status = self.audio_model.get_model_status()
            self.logger.debug("模型状态获取成功 | Model status retrieved successfully")
            return status
        except Exception as e:
            self.logger.error(f"获取模型状态失败: {e} | Failed to get model status: {e}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    def connect_to_external_api(self, api_name: str, api_config: dict):
        """
        连接到外部API | Connect to external API
        
        参数 Parameters:
        api_name: API名称 | API name
        api_config: API配置 | API configuration
        
        返回 Returns:
        连接结果 | Connection result
        """
        try:
            self.logger.info(f"连接到外部API: {api_name} | Connecting to external API: {api_name}")
            result = self.audio_model.connect_to_external_api(api_name, api_config)
            self.logger.info(f"外部API连接完成: {api_name} | External API connection completed: {api_name}")
            return result
        except Exception as e:
            self.logger.error(f"外部API连接失败: {api_name}, 错误: {e} | External API connection failed: {api_name}, error: {e}")
            return {
                "success": False,
                "error": str(e),
                "api_name": api_name
            }
    
    def disconnect_from_external_api(self, api_name: str):
        """
        断开外部API连接 | Disconnect from external API
        
        参数 Parameters:
        api_name: API名称 | API name
        
        返回 Returns:
        断开连接结果 | Disconnection result
        """
        try:
            self.logger.info(f"断开外部API连接: {api_name} | Disconnecting from external API: {api_name}")
            result = self.audio_model.disconnect_from_external_api(api_name)
            self.logger.info(f"外部API连接已断开: {api_name} | External API connection disconnected: {api_name}")
            return result
        except Exception as e:
            self.logger.error(f"断开外部API连接失败: {api_name}, 错误: {e} | Failed to disconnect from external API: {api_name}, error: {e}")
            return {
                "success": False,
                "error": str(e),
                "api_name": api_name
            }
    
    def update_configuration(self, key_path: str, value):
        """
        更新配置 | Update configuration
        
        参数 Parameters:
        key_path: 配置键路径 | Configuration key path
        value: 配置值 | Configuration value
        
        返回 Returns:
        更新结果 | Update result
        """
        try:
            self.logger.info(f"更新配置: {key_path} = {value} | Updating configuration: {key_path} = {value}")
            success = self.config_loader.update_config_value(key_path, value)
            if success:
                self.config_loader.save_config()
                # 重新加载配置到模型
                self.config = self.config_loader.load_config()
                self.audio_model.update_config(self.config)
                self.logger.info(f"配置更新成功: {key_path} | Configuration updated successfully: {key_path}")
                return {"success": True}
            else:
                self.logger.error(f"配置更新失败: {key_path} | Configuration update failed: {key_path}")
                return {"success": False, "error": "配置更新失败"}
        except Exception as e:
            self.logger.error(f"配置更新异常: {key_path}, 错误: {e} | Configuration update exception: {key_path}, error: {e}")
            return {
                "success": False,
                "error": str(e),
                "key_path": key_path
            }
    
    def get_configuration(self, key_path: str = None, default=None):
        """
        获取配置 | Get configuration
        
        参数 Parameters:
        key_path: 配置键路径 | Configuration key path
        default: 默认值 | Default value
        
        返回 Returns:
        配置值 | Configuration value
        """
        try:
            if key_path is None:
                return self.config
            
            value = self.config_loader.get_config_value(key_path, default)
            self.logger.debug(f"获取配置: {key_path} = {value} | Get configuration: {key_path} = {value}")
            return value
        except Exception as e:
            self.logger.error(f"获取配置失败: {key_path}, 错误: {e} | Failed to get configuration: {key_path}, error: {e}")
            return default
    
    def shutdown(self):
        """关闭模型 | Shutdown model"""
        try:
            self.logger.info("正在关闭音频处理模型... | Shutting down audio processing model...")
            # 停止实时处理
            self.stop_realtime_processing()
            # 执行其他清理操作
            self.logger.info("音频处理模型已关闭 | Audio processing model shut down")
            return True
        except Exception as e:
            self.logger.error(f"模型关闭失败: {e} | Model shutdown failed: {e}")
            return False

# 命令行接口 | Command line interface
def main():
    """主函数 | Main function"""
    parser = argparse.ArgumentParser(description='音频处理模型主程序 | Audio Processing Model Main Program')
    parser.add_argument('--config', '-c', type=str, default='sub_models/C_audio/config.yaml',
                       help='配置文件路径 | Configuration file path')
    parser.add_argument('--mode', '-m', type=str, choices=['standalone', 'service'],
                       default='standalone', help='运行模式 | Running mode')
    parser.add_argument('--test', '-t', action='store_true',
                       help='运行测试 | Run tests')
    parser.add_argument('--status', '-s', action='store_true',
                       help='显示模型状态 | Show model status')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='详细输出 | Verbose output')
    
    args = parser.parse_args()
    
    # 设置日志级别
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # 创建主程序实例
    audio_main = AudioModelMain(args.config)
    
    if args.test:
        # 运行测试
        from test_audio_model import AudioModelTester
        tester = AudioModelTester()
        success = tester.run_all_tests()
        sys.exit(0 if success else 1)
    
    elif args.status:
        # 显示模型状态
        status = audio_main.get_model_status()
        print("模型状态 | Model Status:")
        print(f"状态: {status.get('status', 'unknown')}")
        print(f"内存使用: {status.get('memory_usage_mb', 0)} MB")
        print(f"实时处理: {'运行中' if status.get('realtime_processing', False) else '停止'}")
        print(f"外部API: {'启用' if status.get('use_external_api', False) else '禁用'}")
        sys.exit(0)
    
    elif args.mode == 'standalone':
        # 独立模式 - 运行简单的演示
        logger.info("运行独立模式演示 | Running standalone mode demo")
        
        try:
            # 生成测试音频
            import numpy as np
            sample_rate = audio_main.get_configuration("audio.sample_rate", 16000)
            duration = 2.0
            t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
            test_audio = 0.5 * np.sin(2 * np.pi * 440.0 * t)  # 440Hz正弦波
            
            # 测试语音识别
            logger.info("测试语音识别功能 | Testing speech recognition functionality")
            result = audio_main.process_audio(test_audio, AudioProcessingMode.SPEECH_RECOGNITION)
            print(f"语音识别结果: {result}")
            
            # 测试语调分析
            logger.info("测试语调分析功能 | Testing tone analysis functionality")
            result = audio_main.process_audio(test_audio, AudioProcessingMode.TONE_ANALYSIS)
            print(f"语调分析结果: {result}")
            
            # 显示最终状态
            status = audio_main.get_model_status()
            print(f"最终模型状态: {status}")
            
        except Exception as e:
            logger.error(f"独立模式运行失败: {e} | Standalone mode execution failed: {e}")
            sys.exit(1)
        
        finally:
            # 关闭模型
            audio_main.shutdown()
    
    elif args.mode == 'service':
        # 服务模式 - 启动HTTP服务或其他长期运行的服务
        logger.info("启动服务模式 | Starting service mode")
        
        try:
            # 这里可以添加HTTP服务器或其他服务启动代码
            # 目前先保持简单，只显示状态
            status = audio_main.get_model_status()
            logger.info(f"服务模式启动成功，模型状态: {status}")
            
            # 保持运行直到用户中断
            import time
            while True:
                time.sleep(1)
                
        except KeyboardInterrupt:
            logger.info("接收到中断信号，正在关闭服务 | Received interrupt signal, shutting down service")
        
        except Exception as e:
            logger.error(f"服务模式运行失败: {e} | Service mode execution failed: {e}")
        
        finally:
            # 关闭模型
            audio_main.shutdown()
    
    else:
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main()
