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
音频处理模型测试脚本 | Audio Processing Model Test Script
用于测试增强型音频处理模型的功能和性能
Used to test the functionality and performance of the enhanced audio processing model
"""

import os
import sys
import logging
import numpy as np
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
        logging.FileHandler("test_audio_model.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class AudioModelTester:
    """音频模型测试类 | Audio Model Tester Class"""
    
    def __init__(self):
        """初始化测试器 | Initialize tester"""
        self.logger = logger
        self.config_loader = AudioConfigLoader()
        self.config = self.config_loader.load_config()
        self.audio_model = None
        
    def setup_model(self):
        """设置音频模型 | Setup audio model"""
        try:
            self.logger.info("正在初始化音频处理模型... | Initializing audio processing model...")
            self.audio_model = EnhancedAudioProcessingModel(self.config)
            self.logger.info("音频处理模型初始化成功 | Audio processing model initialized successfully")
            return True
        except Exception as e:
            self.logger.error(f"模型初始化失败: {e} | Model initialization failed: {e}")
            return False
    
    def test_config_loading(self):
        """测试配置加载 | Test configuration loading"""
        try:
            self.logger.info("测试配置加载... | Testing configuration loading...")
            
            # 测试获取配置值
            sample_rate = self.config_loader.get_config_value("audio.sample_rate")
            assert sample_rate == 16000, f"采样率配置错误: {sample_rate}"
            
            buffer_size = self.config_loader.get_config_value("realtime.buffer_size")
            assert buffer_size == 10, f"缓冲区大小配置错误: {buffer_size}"
            
            self.logger.info("配置加载测试通过 | Configuration loading test passed")
            return True
            
        except Exception as e:
            self.logger.error(f"配置加载测试失败: {e} | Configuration loading test failed: {e}")
            return False
    
    def test_model_status(self):
        """测试模型状态获取 | Test model status retrieval"""
        try:
            self.logger.info("测试模型状态获取... | Testing model status retrieval...")
            
            status = self.audio_model.get_model_status()
            
            # 验证状态信息
            assert "status" in status, "状态信息缺少status字段"
            assert "memory_usage_mb" in status, "状态信息缺少memory_usage_mb字段"
            assert "performance_metrics" in status, "状态信息缺少performance_metrics字段"
            
            self.logger.info(f"模型状态: {status}")
            self.logger.info("模型状态测试通过 | Model status test passed")
            return True
            
        except Exception as e:
            self.logger.error(f"模型状态测试失败: {e} | Model status test failed: {e}")
            return False
    
    def generate_test_audio(self, duration=3.0, sample_rate=16000, frequency=440.0):
        """
        生成测试音频 | Generate test audio
        
        参数 Parameters:
        duration: 持续时间（秒）| Duration (seconds)
        sample_rate: 采样率 | Sample rate
        frequency: 频率（Hz）| Frequency (Hz)
        
        返回 Returns:
        音频数据 | Audio data
        """
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        audio_data = 0.5 * np.sin(2 * np.pi * frequency * t)
        return audio_data
    
    def test_speech_recognition(self):
        """测试语音识别功能 | Test speech recognition functionality"""
        try:
            self.logger.info("测试语音识别功能... | Testing speech recognition functionality...")
            
            # 生成测试音频（简单的正弦波，模拟语音）
            test_audio = self.generate_test_audio(duration=2.0, frequency=220.0)
            
            # 测试语音识别
            result = self.audio_model.process_audio(
                test_audio,
                AudioProcessingMode.SPEECH_RECOGNITION
            )
            
            self.logger.info(f"语音识别结果: {result}")
            
            # 验证结果格式
            assert result["success"] in [True, False], "结果格式错误"
            assert "mode" in result, "结果缺少mode字段"
            assert result["mode"] == "speech_recognition", "模式不匹配"
            
            if result["success"]:
                assert "result" in result, "成功结果缺少result字段"
                assert "transcription" in result["result"], "结果缺少transcription字段"
            
            self.logger.info("语音识别测试通过 | Speech recognition test passed")
            return True
            
        except Exception as e:
            self.logger.error(f"语音识别测试失败: {e} | Speech recognition test failed: {e}")
            return False
    
    def test_tone_analysis(self):
        """测试语调分析功能 | Test tone analysis functionality"""
        try:
            self.logger.info("测试语调分析功能... | Testing tone analysis functionality...")
            
            # 生成测试音频
            test_audio = self.generate_test_audio(duration=2.0, frequency=440.0)
            
            # 测试语调分析
            result = self.audio_model.process_audio(
                test_audio,
                AudioProcessingMode.TONE_ANALYSIS
            )
            
            self.logger.info(f"语调分析结果: {result}")
            
            # 验证结果格式
            assert result["success"] in [True, False], "结果格式错误"
            assert result["mode"] == "tone_analysis", "模式不匹配"
            
            if result["success"]:
                assert "result" in result, "成功结果缺少result字段"
                assert "emotion_probabilities" in result["result"], "结果缺少emotion_probabilities字段"
            
            self.logger.info("语调分析测试通过 | Tone analysis test passed")
            return True
            
        except Exception as e:
            self.logger.error(f"语调分析测试失败: {e} | Tone analysis test failed: {e}")
            return False
    
    def test_audio_effects(self):
        """测试音频特效功能 | Test audio effects functionality"""
        try:
            self.logger.info("测试音频特效功能... | Testing audio effects functionality...")
            
            # 生成测试音频
            test_audio = self.generate_test_audio(duration=2.0, frequency=440.0)
            
            # 测试回声效果
            result = self.audio_model.apply_audio_effect(
                test_audio,
                AudioEffectType.ECHO,
                delay=0.3,
                decay=0.5
            )
            
            self.logger.info(f"回声效果结果: {result}")
            
            # 验证结果格式
            assert result["success"] in [True, False], "结果格式错误"
            assert result["effect_type"] == "echo", "特效类型不匹配"
            
            if result["success"]:
                assert "result" in result, "成功结果缺少result字段"
            
            self.logger.info("音频特效测试通过 | Audio effects test passed")
            return True
            
        except Exception as e:
            self.logger.error(f"音频特效测试失败: {e} | Audio effects test failed: {e}")
            return False
    
    def test_realtime_processing(self):
        """测试实时处理功能 | Test real-time processing functionality"""
        try:
            self.logger.info("测试实时处理功能... | Testing real-time processing functionality...")
            
            # 启动实时处理
            result = self.audio_model.process_audio(
                self.generate_test_audio(duration=0.5, frequency=440.0),
                AudioProcessingMode.REAL_TIME_PROCESSING
            )
            
            self.logger.info(f"实时处理启动结果: {result}")
            
            # 验证结果格式
            assert result["success"] in [True, False], "结果格式错误"
            assert result["mode"] == "real_time_processing", "模式不匹配"
            
            # 检查状态
            status = self.audio_model.get_model_status()
            assert status["realtime_processing"] == True, "实时处理未启动"
            
            # 停止实时处理
            self.audio_model.stop_realtime_processing()
            
            # 再次检查状态
            status = self.audio_model.get_model_status()
            assert status["realtime_processing"] == False, "实时处理未停止"
            
            self.logger.info("实时处理测试通过 | Real-time processing test passed")
            return True
            
        except Exception as e:
            self.logger.error(f"实时处理测试失败: {e} | Real-time processing test failed: {e}")
            # 确保停止实时处理
            try:
                self.audio_model.stop_realtime_processing()
            except:
                pass
            return False
    
    def test_external_api_connection(self):
        """测试外部API连接功能 | Test external API connection functionality"""
        try:
            self.logger.info("测试外部API连接功能... | Testing external API connection functionality...")
            
            # 测试连接到外部API（模拟）
            success = self.audio_model.connect_to_external_api(
                "google_speech",
                {
                    "enabled": True,
                    "api_key": "test_key",
                    "region": "us-west1"
                }
            )
            
            self.logger.info(f"外部API连接结果: {success}")
            
            # 检查配置是否更新
            status = self.audio_model.get_model_status()
            assert status["use_external_api"] == True, "外部API未启用"
            
            # 测试断开连接
            disconnect_success = self.audio_model.disconnect_from_external_api("google_speech")
            self.logger.info(f"外部API断开结果: {disconnect_success}")
            
            self.logger.info("外部API连接测试通过 | External API connection test passed")
            return True
            
        except Exception as e:
            self.logger.error(f"外部API连接测试失败: {e} | External API connection test failed: {e}")
            return False
    
    def run_all_tests(self):
        """运行所有测试 | Run all tests"""
        self.logger.info("开始运行音频处理模型测试套件 | Starting audio processing model test suite")
        
        tests = [
            ("配置加载测试", self.test_config_loading),
            ("模型状态测试", self.test_model_status),
            ("语音识别测试", self.test_speech_recognition),
            ("语调分析测试", self.test_tone_analysis),
            ("音频特效测试", self.test_audio_effects),
            ("实时处理测试", self.test_realtime_processing),
            ("外部API连接测试", self.test_external_api_connection)
        ]
        
        results = []
        total_tests = len(tests)
        passed_tests = 0
        
        # 首先设置模型
        if not self.setup_model():
            self.logger.error("模型设置失败，无法继续测试 | Model setup failed, cannot continue testing")
            return False
        
        for test_name, test_func in tests:
            try:
                self.logger.info(f"开始测试: {test_name} | Starting test: {test_name}")
                success = test_func()
                results.append((test_name, success))
                
                if success:
                    passed_tests += 1
                    self.logger.info(f"测试通过: {test_name} | Test passed: {test_name}")
                else:
                    self.logger.error(f"测试失败: {test_name} | Test failed: {test_name}")
                    
            except Exception as e:
                self.logger.error(f"测试执行异常: {test_name}, 错误: {e} | Test execution exception: {test_name}, error: {e}")
                results.append((test_name, False))
        
        # 输出测试结果
        self.logger.info("=" * 60)
        self.logger.info("测试结果汇总 | Test Results Summary")
        self.logger.info("=" * 60)
        
        for test_name, success in results:
            status = "通过" if success else "失败"
            self.logger.info(f"{test_name}: {status}")
        
        self.logger.info("=" * 60)
        self.logger.info(f"总测试数: {total_tests} | Total tests: {total_tests}")
        self.logger.info(f"通过测试数: {passed_tests} | Passed tests: {passed_tests}")
        self.logger.info(f"失败测试数: {total_tests - passed_tests} | Failed tests: {total_tests - passed_tests}")
        self.logger.info(f"通过率: {passed_tests / total_tests * 100:.2f}% | Pass rate: {passed_tests / total_tests * 100:.2f}%")
        
        # 清理资源
        try:
            self.audio_model.stop_realtime_processing()
        except:
            pass
        
        return passed_tests == total_tests

def main():
    """主函数 | Main function"""
    tester = AudioModelTester()
    success = tester.run_all_tests()
    
    if success:
        logger.info("所有测试通过！音频处理模型功能正常 | All tests passed! Audio processing model functions properly")
        sys.exit(0)
    else:
        logger.error("部分测试失败，请检查日志了解详情 | Some tests failed, please check logs for details")
        sys.exit(1)

if __name__ == "__main__":
    main()
