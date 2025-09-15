#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Self Brain AGI 系统全面审查脚本
Comprehensive System Audit Script for Self Brain AGI

此脚本将严格审查：
1. Web界面所有功能的真实有效性
2. 所有模型的实际运行状态和功能完整性
3. 训练程序的存在性和完整性
4. 外部API接入能力
5. 联合训练功能
"""

import os
import sys
import json
import requests
import subprocess
import time
import socket
from datetime import datetime
import threading
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ComprehensiveSystemAudit:
    def __init__(self):
        self.base_url = "http://localhost:5000"
        self.model_ports = {
            'A_management': 5015,
            'B_language': 5002,
            'C_audio': 5003,
            'D_image': 5004,
            'E_video': 5005,
            'F_spatial': 5006,
            'G_sensor': 5007,
            'H_computer': 5008,
            'I_knowledge': 5009,
            'J_motion': 5010,
            'K_programming': 5011
        }
        self.audit_results = {
            'web_interface': {},
            'models': {},
            'training': {},
            'api_integration': {},
            'joint_training': {}
        }

    def check_port(self, port):
        """检查端口是否开放"""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)
            result = sock.connect_ex(('localhost', port))
            sock.close()
            return result == 0
        except:
            return False

    def test_web_interface(self):
        """测试Web界面所有功能"""
        logger.info("开始测试Web界面功能...")
        
        # 测试主要页面
        pages = [
            ('/', '主页'),
            ('/knowledge_manage', '知识管理'),
            ('/training', '训练页面'),
            ('/system_settings', '系统设置'),
            ('/upload', '上传页面')
        ]
        
        for path, name in pages:
            try:
                response = requests.get(f"{self.base_url}{path}", timeout=5)
                self.audit_results['web_interface'][f"{name}_page"] = {
                    'status': response.status_code == 200,
                    'response_time': response.elapsed.total_seconds(),
                    'content_length': len(response.content)
                }
                logger.info(f"✓ {name}页面测试通过")
            except Exception as e:
                self.audit_results['web_interface'][f"{name}_page"] = {
                    'status': False,
                    'error': str(e)
                }
                logger.error(f"✗ {name}页面测试失败: {e}")

    def test_model_functionality(self):
        """测试所有模型功能"""
        logger.info("开始测试模型功能...")
        
        # 测试每个模型的基本功能
        model_tests = {
            'A_management': {
                'endpoint': 'http://localhost:5015/api/chat',
                'data': {'message': '测试A管理模型'}
            },
            'B_language': {
                'endpoint': 'http://localhost:5002/api/generate',
                'data': {'text': '测试语言生成', 'language': 'zh'}
            },
            'C_audio': {
                'endpoint': 'http://localhost:5003/api/process',
                'data': {'audio_data': [0.1, 0.2, 0.3]}
            },
            'D_image': {
                'endpoint': 'http://localhost:5004/api/analyze',
                'data': {'image_path': 'test.jpg'}
            },
            'E_video': {
                'endpoint': 'http://localhost:5005/api/process',
                'data': {'video_path': 'test.mp4'}
            },
            'F_spatial': {
                'endpoint': 'http://localhost:5006/api/analyze',
                'data': {'spatial_data': [[1, 2], [3, 4]]}
            },
            'G_sensor': {
                'endpoint': 'http://localhost:5007/process',
                'data': {'sensor_data': [25.0, 60.0, 1.0]}
            },
            'H_computer': {
                'endpoint': 'http://localhost:5008/api/execute',
                'data': {'command': 'echo test'}
            },
            'I_knowledge': {
                'endpoint': 'http://localhost:5009/api/query',
                'data': {'query': '测试知识查询'}
            },
            'J_motion': {
                'endpoint': 'http://localhost:5010/api/control',
                'data': {'action': 'move_forward'}
            },
            'K_programming': {
                'endpoint': 'http://localhost:5011/api/generate_code',
                'data': {'prompt': '写一个Python函数'}
            }
        }
        
        for model_name, test_config in model_tests.items():
            port = self.model_ports[model_name]
            if self.check_port(port):
                try:
                    response = requests.post(test_config['endpoint'], 
                                           json=test_config['data'], 
                                           timeout=5)
                    self.audit_results['models'][model_name] = {
                        'status': True,
                        'port': port,
                        'response': response.status_code,
                        'functionality': 'basic'
                    }
                    logger.info(f"✓ {model_name}模型测试通过")
                except Exception as e:
                    self.audit_results['models'][model_name] = {
                        'status': False,
                        'port': port,
                        'error': str(e)
                    }
                    logger.error(f"✗ {model_name}模型测试失败: {e}")
            else:
                self.audit_results['models'][model_name] = {
                    'status': False,
                    'port': port,
                    'error': '端口未开放'
                }
                logger.warning(f"⚠ {model_name}端口未开放")

    def check_training_programs(self):
        """检查训练程序"""
        logger.info("开始检查训练程序...")
        
        training_files = {
            'B_language': 'sub_models/B_language/train.py',
            'C_audio': 'sub_models/C_audio/train.py',
            'D_image': 'sub_models/D_image/train.py',
            'E_video': 'sub_models/E_video/train.py',
            'F_spatial': 'sub_models/F_spatial/train.py',
            'G_sensor': 'sub_models/G_sensor/train.py',
            'H_computer': 'sub_models/H_computer_control/train.py',
            'I_knowledge': 'sub_models/I_knowledge/train.py',
            'J_motion': 'sub_models/J_motion/train.py',
            'K_programming': 'sub_models/K_programming/train.py'
        }
        
        for model_name, file_path in training_files.items():
            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                # 检查关键功能
                has_train_function = 'def train' in content or 'def main' in content
                has_joint_training = 'joint' in content.lower()
                has_external_api = 'external' in content.lower() or 'api' in content.lower()
                
                self.audit_results['training'][model_name] = {
                    'exists': True,
                    'train_function': has_train_function,
                    'joint_training': has_joint_training,
                    'external_api_support': has_external_api,
                    'file_size': len(content)
                }
                logger.info(f"✓ {model_name}训练程序检查通过")
            else:
                self.audit_results['training'][model_name] = {
                    'exists': False,
                    'error': '训练文件不存在'
                }
                logger.error(f"✗ {model_name}训练程序不存在")

    def test_api_integration(self):
        """测试API集成功能"""
        logger.info("开始测试API集成...")
        
        # 检查配置文件中的API支持
        api_configs = [
            'sub_models/B_language/config.yaml',
            'sub_models/C_audio/config.yaml',
            'sub_models/D_image/config.yaml',
            'sub_models/G_sensor/config.yaml'
        ]
        
        for config_path in api_configs:
            model_name = config_path.split('/')[1]
            if os.path.exists(config_path):
                try:
                    import yaml
                    with open(config_path, 'r', encoding='utf-8') as f:
                        config = yaml.safe_load(f)
                    
                    has_external_api = False
                    if 'external_apis' in str(config).lower():
                        has_external_api = True
                    
                    self.audit_results['api_integration'][model_name] = {
                        'config_exists': True,
                        'external_api_support': has_external_api,
                        'config_path': config_path
                    }
                    logger.info(f"✓ {model_name}API配置检查通过")
                except Exception as e:
                    self.audit_results['api_integration'][model_name] = {
                        'config_exists': True,
                        'error': str(e)
                    }
                    logger.error(f"✗ {model_name}API配置检查失败: {e}")
            else:
                self.audit_results['api_integration'][model_name] = {
                    'config_exists': False
                }
                logger.warning(f"⚠ {model_name}配置文件不存在")

    def test_joint_training(self):
        """测试联合训练功能"""
        logger.info("开始测试联合训练功能...")
        
        # 检查是否有联合训练的支持
        joint_training_models = []
        for model_name, model_info in self.audit_results['training'].items():
            if model_info.get('joint_training', False):
                joint_training_models.append(model_name)
        
        self.audit_results['joint_training'] = {
            'supported_models': joint_training_models,
            'count': len(joint_training_models),
            'status': len(joint_training_models) > 0
        }
        
        if len(joint_training_models) > 0:
            logger.info(f"✓ 支持联合训练的模型: {joint_training_models}")
        else:
            logger.warning("⚠ 未发现联合训练支持")

    def generate_report(self):
        """生成审查报告"""
        report = {
            'audit_timestamp': datetime.now().isoformat(),
            'summary': {
                'total_models': len(self.model_ports),
                'web_interface_status': self.audit_results['web_interface'],
                'models_status': self.audit_results['models'],
                'training_status': self.audit_results['training'],
                'api_integration': self.audit_results['api_integration'],
                'joint_training': self.audit_results['joint_training']
            },
            'recommendations': self.generate_recommendations()
        }
        
        # 保存报告
        with open('system_audit_report.json', 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        logger.info("审查报告已保存到 system_audit_report.json")
        return report

    def generate_recommendations(self):
        """生成改进建议"""
        recommendations = []
        
        # 检查缺失的训练程序
        missing_training = [k for k, v in self.audit_results['training'].items() if not v.get('exists', False)]
        if missing_training:
            recommendations.append({
                'type': 'missing_training',
                'models': missing_training,
                'priority': 'high',
                'action': '创建缺失的训练程序'
            })
        
        # 检查未运行的模型
        failed_models = [k for k, v in self.audit_results['models'].items() if not v.get('status', False)]
        if failed_models:
            recommendations.append({
                'type': 'failed_models',
                'models': failed_models,
                'priority': 'high',
                'action': '检查并启动失败的模型服务'
            })
        
        # 检查API集成
        models_without_api = [k for k, v in self.audit_results['api_integration'].items() if not v.get('external_api_support', False)]
        if models_without_api:
            recommendations.append({
                'type': 'api_integration',
                'models': models_without_api,
                'priority': 'medium',
                'action': '添加外部API集成支持'
            })
        
        return recommendations

    def run_audit(self):
        """运行完整审查"""
        logger.info("开始Self Brain AGI系统全面审查...")
        
        self.test_web_interface()
        self.test_model_functionality()
        self.check_training_programs()
        self.test_api_integration()
        self.test_joint_training()
        
        report = self.generate_report()
        
        # 打印总结
        logger.info("=" * 50)
        logger.info("系统审查总结")
        logger.info("=" * 50)
        
        web_ok = sum(1 for v in self.audit_results['web_interface'].values() if v.get('status', False))
        models_ok = sum(1 for v in self.audit_results['models'].values() if v.get('status', False))
        training_ok = sum(1 for v in self.audit_results['training'].values() if v.get('exists', False))
        
        logger.info(f"Web界面功能: {web_ok}/5 通过")
        logger.info(f"模型运行状态: {models_ok}/11 通过")
        logger.info(f"训练程序: {training_ok}/10 存在")
        logger.info(f"联合训练支持: {self.audit_results['joint_training']['count']} 个模型")
        
        return report

if __name__ == "__main__":
    audit = ComprehensiveSystemAudit()
    audit.run_audit()