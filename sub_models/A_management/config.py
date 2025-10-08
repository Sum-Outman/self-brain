# Copyright 2025 The AI Management System Authors
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
管理模型配置文件，包含模型、训练和评估的配置参数
"""

import os
from datetime import datetime
import torch

# 基础配置
class BaseConfig:
    def __init__(self):
        # 项目根目录
        self.root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # 设备配置
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 输出目录
        self.output_dir = os.path.join(self.root_dir, 'outputs')
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 时间戳
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 日志配置
        self.log_dir = os.path.join(self.output_dir, 'logs')
        os.makedirs(self.log_dir, exist_ok=True)
        
        # 数据配置
        self.data_dir = os.path.join(self.root_dir, 'data')
        os.makedirs(self.data_dir, exist_ok=True)
        
        # 模型配置
        self.model_dir = os.path.join(self.output_dir, 'models')
        os.makedirs(self.model_dir, exist_ok=True)
        
        # 临时目录
        self.temp_dir = os.path.join(self.output_dir, 'temp')
        os.makedirs(self.temp_dir, exist_ok=True)

# 模型配置
class ModelConfig(BaseConfig):
    def __init__(self):
        super().__init__()
        
        # 模型基本配置
        self.model_name = "management_model"
        self.model_version = "1.0"
        self.model_type = "ensemble"
        
        # 输入输出配置
        self.input_dim = 768  # 输入特征维度
        self.output_dim = 4   # 输出策略类别数量
        
        # 情感配置
        self.emotion_dim = 7  # 情感类别数量 (neutral, joy, sadness, anger, fear, surprise, disgust)
        
        # 神经网络配置
        self.hidden_dims = [512, 256]  # 隐藏层维度
        self.dropout_rate = 0.3        # Dropout率
        self.activation = "relu"       # 激活函数
        
        # 优化器配置
        self.learning_rate = 1e-4      # 学习率
        self.weight_decay = 1e-5       # 权重衰减
        self.beta1 = 0.9               # Adam优化器参数
        self.beta2 = 0.999             # Adam优化器参数
        
        # 集成学习配置
        self.use_ensemble = True       # 是否使用集成学习
        self.ensemble_method = "weighted_average"  # 集成方法 (weighted_average, voting)
        self.sub_model_weights = {     # 下属模型权重
            "B_language": 0.4,
            "B_vision": 0.3,
            "B_multimodal": 0.3
        }
        
        # 情感分析配置
        self.use_emotion_analysis = True      # 是否使用情感分析
        self.emotion_weight = 0.3             # 情感在最终决策中的权重
        self.emotion_intensity_threshold = 0.5  # 情感强度阈值
        
        # 缓存配置
        self.use_cache = True        # 是否使用缓存
        self.cache_size = 1000       # 缓存大小
        self.cache_ttl = 3600        # 缓存过期时间（秒）
        
        # 并行计算配置
        self.use_parallel = False    # 是否使用并行计算
        self.num_workers = 4         # 并行工作线程数
        
        # 下属模型配置路径
        self.sub_model_configs = {
            "B_language": os.path.join(self.root_dir, 'sub_models', 'B_language', 'config.py'),
            "B_vision": os.path.join(self.root_dir, 'sub_models', 'B_vision', 'config.py'),
            "B_multimodal": os.path.join(self.root_dir, 'sub_models', 'B_multimodal', 'config.py')
        }
        
        # 下属模型默认路径
        self.sub_model_default_paths = {
            "B_language": os.path.join(self.root_dir, 'sub_models', 'B_language', 'model.py'),
            "B_vision": os.path.join(self.root_dir, 'sub_models', 'B_vision', 'model.py'),
            "B_multimodal": os.path.join(self.root_dir, 'sub_models', 'B_multimodal', 'model.py')
        }

# 训练配置
class TrainConfig(BaseConfig):
    def __init__(self):
        super().__init__()
        
        # 数据集配置
        self.train_data_path = os.path.join(self.data_dir, 'train_data.json')
        self.val_data_path = os.path.join(self.data_dir, 'val_data.json')
        self.test_data_path = os.path.join(self.data_dir, 'test_data.json')
        self.dataset_split = [0.7, 0.15, 0.15]  # 训练集、验证集、测试集的划分比例
        
        # 训练参数配置
        self.batch_size = 32         # 批大小
        self.epochs = 100            # 训练轮数
        self.shuffle = True          # 是否打乱数据
        self.drop_last = False       # 是否丢弃最后一个不完整的批次
        self.num_workers = 4         # 数据加载器工作线程数
        
        # 优化器配置
        self.optimizer = "AdamW"       # 优化器类型 (Adam, AdamW, SGD)
        self.learning_rate = 1e-4    # 初始学习率
        self.weight_decay = 1e-5     # 权重衰减
        
        # 学习率调度器配置
        self.use_scheduler = True    # 是否使用学习率调度器
        self.scheduler = "ReduceLROnPlateau"  # 调度器类型
        self.scheduler_patience = 5  # 学习率调整的耐心值
        self.scheduler_factor = 0.5  # 学习率调整因子
        self.min_learning_rate = 1e-6  # 最小学习率
        
        # 损失函数配置
        self.loss_function = "cross_entropy"  # 损失函数类型
        self.strategy_weight = 0.7   # 策略预测的损失权重
        self.emotion_weight = 0.3    # 情感预测的损失权重
        
        # 早停配置
        self.use_early_stopping = True  # 是否使用早停
        self.early_stopping_patience = 10  # 早停耐心值
        self.early_stopping_metric = "val_combined_score"  # 早停监控指标
        
        # 梯度裁剪配置
        self.use_grad_clip = True    # 是否使用梯度裁剪
        self.grad_clip_max_norm = 1.0  # 梯度裁剪的最大范数
        
        # 检查点配置
        self.save_checkpoints = True  # 是否保存检查点
        self.checkpoint_dir = os.path.join(self.output_dir, 'checkpoints')
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.checkpoint_interval = 1  # 检查点保存间隔（轮次）
        self.save_best_only = True    # 是否只保存最佳模型
        
        # 日志配置
        self.log_interval = 10       # 日志打印间隔（批次）
        self.use_tensorboard = False  # 是否使用TensorBoard
        self.tensorboard_dir = os.path.join(self.output_dir, 'tensorboard')
        os.makedirs(self.tensorboard_dir, exist_ok=True)
        
        # 评估配置
        self.evaluate_during_training = True  # 是否在训练期间进行评估
        self.evaluation_interval = 1          # 评估间隔（轮次）
        
        # 模型保存配置
        self.save_final_model = True   # 是否保存最终模型
        self.final_model_path = os.path.join(self.model_dir, 'final_model.pth')
        
        # 训练报告配置
        self.save_training_report = True  # 是否保存训练报告
        self.training_report_path = os.path.join(self.output_dir, 'training_report.json')

# 评估配置
class EvalConfig(BaseConfig):
    def __init__(self):
        super().__init__()
        
        # 评估数据配置
        self.eval_data_path = os.path.join(self.data_dir, 'test_data.json')
        self.batch_size = 64          # 评估批大小
        self.shuffle = False          # 是否打乱评估数据
        self.num_workers = 4          # 评估数据加载器工作线程数
        
        # 评估指标配置
        self.metrics = [              # 要计算的评估指标
            "accuracy",
            "precision",
            "recall",
            "f1_score",
            "confusion_matrix"
        ]
        
        # 情感标签配置
        self.emotion_labels = [       # 情感标签列表
            "neutral",
            "joy",
            "sadness",
            "anger",
            "fear",
            "surprise",
            "disgust"
        ]
        
        # 策略标签配置
        self.strategy_labels = [      # 策略标签列表
            "strategy_0",
            "strategy_1",
            "strategy_2",
            "strategy_3"
        ]
        
        # 结果保存配置
        self.save_results = True      # 是否保存评估结果
        self.results_dir = os.path.join(self.output_dir, 'evaluation_results')
        os.makedirs(self.results_dir, exist_ok=True)
        
        # 可视化配置
        self.save_visualizations = True  # 是否保存可视化结果
        self.visualizations_dir = os.path.join(self.output_dir, 'visualizations')
        os.makedirs(self.visualizations_dir, exist_ok=True)
        
        # 混淆矩阵配置
        self.plot_confusion_matrix = True  # 是否绘制混淆矩阵
        self.confusion_matrix_normalize = False  # 是否归一化混淆矩阵
        
        # 指标对比图配置
        self.plot_metrics_comparison = True  # 是否绘制指标对比图
        
        # 下属模型评估配置
        self.evaluate_sub_models = True  # 是否评估下属模型
        
        # 集成有效性分析配置
        self.analyze_integration_effectiveness = True  # 是否分析集成有效性
        
        # 错误分析配置
        self.perform_error_analysis = True  # 是否进行错误分析
        self.max_error_samples = 200        # 最大错误样本数
        
        # 模型行为分析配置
        self.analyze_model_behavior = True  # 是否分析模型行为
        
        # 评估报告配置
        self.generate_evaluation_report = True  # 是否生成评估报告
        self.evaluation_report_path = os.path.join(self.results_dir, 'evaluation_report.json')

# 合并配置类
class Config:
    def __init__(self):
        # 初始化各子配置
        self.base = BaseConfig()
        self.model = ModelConfig()
        self.train = TrainConfig()
        self.eval = EvalConfig()
        
        # 同步目录配置
        self._sync_directories()
        
    def _sync_directories(self):
        """
        同步各子配置中的目录设置，确保它们保持一致
        """
        # 同步根目录
        root_dir = self.base.root_dir
        
        # 同步输出目录
        output_dir = os.path.join(root_dir, 'outputs')
        
        # 更新所有子配置中的目录
        self.base.root_dir = root_dir
        self.base.output_dir = output_dir
        self.model.root_dir = root_dir
        self.model.output_dir = output_dir
        self.train.root_dir = root_dir
        self.train.output_dir = output_dir
        self.eval.root_dir = root_dir
        self.eval.output_dir = output_dir
        
        # 重新创建目录
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(self.base.log_dir, exist_ok=True)
        os.makedirs(self.base.data_dir, exist_ok=True)
        os.makedirs(self.base.model_dir, exist_ok=True)
        os.makedirs(self.train.checkpoint_dir, exist_ok=True)
        os.makedirs(self.eval.results_dir, exist_ok=True)
        os.makedirs(self.eval.visualizations_dir, exist_ok=True)
    
    def save(self, config_path=None):
        """
        保存配置到文件
        
        参数:
            config_path: 配置文件保存路径
        """
        if config_path is None:
            config_path = os.path.join(self.base.output_dir, f'config_{self.base.timestamp}.json')
        
        # 构建配置字典
        config_dict = {
            'base': {
                'root_dir': self.base.root_dir,
                'output_dir': self.base.output_dir,
                'timestamp': self.base.timestamp
            },
            'model': {
                'model_name': self.model.model_name,
                'model_version': self.model.model_version,
                'input_dim': self.model.input_dim,
                'output_dim': self.model.output_dim,
                'emotion_dim': self.model.emotion_dim,
                'hidden_dims': self.model.hidden_dims,
                'dropout_rate': self.model.dropout_rate,
                'activation': self.model.activation,
                'use_ensemble': self.model.use_ensemble,
                'ensemble_method': self.model.ensemble_method,
                'sub_model_weights': self.model.sub_model_weights,
                'use_emotion_analysis': self.model.use_emotion_analysis,
                'emotion_weight': self.model.emotion_weight
            },
            'train': {
                'batch_size': self.train.batch_size,
                'epochs': self.train.epochs,
                'learning_rate': self.train.learning_rate,
                'weight_decay': self.train.weight_decay,
                'optimizer': self.train.optimizer,
                'use_scheduler': self.train.use_scheduler,
                'scheduler': self.train.scheduler,
                'use_early_stopping': self.train.use_early_stopping,
                'early_stopping_patience': self.train.early_stopping_patience,
                'checkpoint_dir': self.train.checkpoint_dir
            },
            'eval': {
                'batch_size': self.eval.batch_size,
                'metrics': self.eval.metrics,
                'emotion_labels': self.eval.emotion_labels,
                'strategy_labels': self.eval.strategy_labels,
                'save_results': self.eval.save_results,
                'results_dir': self.eval.results_dir
            }
        }
        
        # 保存配置到文件
        with open(config_path, 'w', encoding='utf-8') as f:
            import json
            json.dump(config_dict, f, indent=2, ensure_ascii=False)
        
        print(f"Configuration saved to {config_path}")
        
        return config_path
    
    @staticmethod
    def load(config_path):
        """
        从文件加载配置
        
        参数:
            config_path: 配置文件路径
        
        返回:
            配置实例
        """
        # 检查文件是否存在
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        # 加载配置文件
        with open(config_path, 'r', encoding='utf-8') as f:
            import json
            config_dict = json.load(f)
        
        # 创建配置实例
        config = Config()
        
        # 更新配置
        if 'base' in config_dict:
            base_config = config_dict['base']
            if 'root_dir' in base_config:
                config.base.root_dir = base_config['root_dir']
            if 'output_dir' in base_config:
                config.base.output_dir = base_config['output_dir']
        
        if 'model' in config_dict:
            model_config = config_dict['model']
            for key, value in model_config.items():
                if hasattr(config.model, key):
                    setattr(config.model, key, value)
        
        if 'train' in config_dict:
            train_config = config_dict['train']
            for key, value in train_config.items():
                if hasattr(config.train, key):
                    setattr(config.train, key, value)
        
        if 'eval' in config_dict:
            eval_config = config_dict['eval']
            for key, value in eval_config.items():
                if hasattr(config.eval, key):
                    setattr(config.eval, key, value)
        
        # 同步目录
        config._sync_directories()
        
        print(f"Configuration loaded from {config_path}")
        
        return config

# 创建默认配置实例
def get_default_config():
    """
    获取默认配置
    
    返回:
        默认配置实例
    """
    return Config()

# 创建开发环境配置
def get_dev_config():
    """
    获取开发环境配置
    
    返回:
        开发环境配置实例
    """
    config = Config()
    
    # 开发环境特有的配置
    config.train.batch_size = 16  # 开发环境使用较小的批大小
    config.train.epochs = 10      # 开发环境使用较少的训练轮数
    config.train.log_interval = 1 # 开发环境增加日志频率
    config.eval.batch_size = 32   # 开发环境使用较小的评估批大小
    
    # 快速测试配置
    config.train.early_stopping_patience = 3
    config.train.evaluation_interval = 1
    
    return config

# 创建生产环境配置
def get_prod_config():
    """
    获取生产环境配置
    
    返回:
        生产环境配置实例
    """
    config = Config()
    
    # 生产环境特有的配置
    config.train.batch_size = 64   # 生产环境使用较大的批大小
    config.train.epochs = 200      # 生产环境使用较多的训练轮数
    config.train.log_interval = 50 # 生产环境减少日志频率
    config.eval.batch_size = 128   # 生产环境使用较大的评估批大小
    
    # 生产环境优化配置
    config.model.dropout_rate = 0.4  # 生产环境增加dropout以防止过拟合
    config.train.weight_decay = 5e-5 # 生产环境增加权重衰减
    
    # 生产环境保存配置
    config.train.save_checkpoints = True
    config.train.save_best_only = True
    config.train.save_training_report = True
    
    return config

# 创建测试环境配置
def get_test_config():
    """
    获取测试环境配置
    
    返回:
        测试环境配置实例
    """
    config = Config()
    
    # 测试环境特有的配置
    config.train.batch_size = 8    # 测试环境使用很小的批大小
    config.train.epochs = 2        # 测试环境只使用极少的训练轮数
    config.eval.batch_size = 16    # 测试环境使用很小的评估批大小
    
    # 快速测试配置
    config.train.early_stopping_patience = 1
    config.train.evaluation_interval = 1
    
    return config

# 配置验证函数
def validate_config(config):
    """
    验证配置的有效性
    
    参数:
        config: 配置实例
    
    返回:
        是否有效
    """
    # 检查必要的目录是否存在
    required_dirs = [
        config.base.root_dir,
        config.base.data_dir,
        config.base.model_dir,
        config.train.checkpoint_dir,
        config.eval.results_dir
    ]
    
    for dir_path in required_dirs:
        if not os.path.exists(dir_path):
            try:
                os.makedirs(dir_path, exist_ok=True)
            except Exception as e:
                print(f"Failed to create directory {dir_path}: {str(e)}")
                return False
    
    # 检查配置参数的有效性
    if config.model.input_dim <= 0:
        print("Invalid input_dim: must be positive")
        return False
    
    if config.model.output_dim <= 0:
        print("Invalid output_dim: must be positive")
        return False
    
    if config.model.emotion_dim <= 0:
        print("Invalid emotion_dim: must be positive")
        return False
    
    if config.train.batch_size <= 0:
        print("Invalid batch_size: must be positive")
        return False
    
    if config.train.epochs <= 0:
        print("Invalid epochs: must be positive")
        return False
    
    if config.train.learning_rate <= 0:
        print("Invalid learning_rate: must be positive")
        return False
    
    # 检查模型权重的有效性
    if config.model.use_ensemble:
        total_weight = sum(config.model.sub_model_weights.values())
        if abs(total_weight - 1.0) > 1e-6:
            print(f"Warning: Sub-model weights do not sum to 1.0 (sum: {total_weight})")
    
    # 配置有效
    return True

# 配置示例
if __name__ == "__main__":
    # 创建默认配置
    config = get_default_config()
    
    # 验证配置
    is_valid = validate_config(config)
    print(f"Configuration is valid: {is_valid}")
    
    # 保存配置
    if is_valid:
        config_path = config.save()
        print(f"Configuration saved to {config_path}")
        
        # 加载配置
        loaded_config = Config.load(config_path)
        print("Configuration loaded successfully")
    
    # 测试不同环境的配置
    dev_config = get_dev_config()
    prod_config = get_prod_config()
    test_config = get_test_config()
    
    print(f"\nDev config epochs: {dev_config.train.epochs}")
    print(f"Prod config epochs: {prod_config.train.epochs}")
    print(f"Test config epochs: {test_config.train.epochs}")