<<<<<<< HEAD
# 将manager_model目录标记为Python包

# 从training_control模块导出TrainingController类和get_training_controller函数
from .training_control import TrainingController, get_training_controller

__all__ = ['TrainingController', 'get_training_controller']
=======
# Manager Model Package Initialization

# Import and expose key components
from .data_bus import get_data_bus, DataBus, data_bus_bp
from .emotion_engine import get_emotion_engine, EmotionEngine, emotion_bp
from .training_control import get_training_controller, TrainingController, training_control_bp

# Package version
__version__ = '1.0.0'

# Initialize package components
def init_manager_model():
    """Initialize the manager model package"""
    # Ensure data bus is initialized
    get_data_bus()
    # Ensure emotion engine is initialized
    get_emotion_engine()
    # Ensure training controller is initialized
    get_training_controller()
    
    # Register blueprints
    blueprints = [data_bus_bp, emotion_bp, training_control_bp]
    
    return blueprints
>>>>>>> 55541e2569d492f61ad4c096b6721db4fe055a13
