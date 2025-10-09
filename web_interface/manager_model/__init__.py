# 将manager_model目录标记为Python包

# 从training_control模块导出TrainingController类和get_training_controller函数
from .training_control import TrainingController, get_training_controller

__all__ = ['TrainingController', 'get_training_controller']