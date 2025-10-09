#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
生成管理模型的示例训练数据
"""
import os
import json
import numpy as np
import random
from datetime import datetime

# 设置随机种子以保证结果可复现
random.seed(42)
np.random.seed(42)

# 数据目录
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')

# 情感标签映射
strategy_mapping = {
    0: "sequential",
    1: "parallel",
    2: "hierarchical",
    3: "adaptive"
}

# 情感标签映射
emotion_mapping = {
    0: "happy",
    1: "sad",
    2: "angry",
    3: "fear",
    4: "surprise",
    5: "disgust",
    6: "neutral"
}

# 生成示例特征向量
def generate_feature_vector(feature_dim=50):
    """生成模拟的输入特征向量"""
    # 创建一个基础特征向量，使用正态分布
    base_features = np.random.randn(feature_dim)
    
    # 为了使数据更有意义，我们可以添加一些特定模式
    # 例如，某些维度表示任务复杂度、紧急程度、资源可用性等
    task_complexity = random.uniform(0, 1)  # 任务复杂度 (0-1)
    task_urgency = random.uniform(0, 1)     # 任务紧急程度 (0-1)
    resource_availability = random.uniform(0, 1)  # 资源可用性 (0-1)
    team_size = random.randint(1, 20)       # 团队规模 (1-20)
    
    # 将这些特征添加到基础特征中
    feature_vector = np.zeros(feature_dim)
    feature_vector[:4] = [task_complexity, task_urgency, resource_availability, team_size]
    feature_vector[4:] = base_features[4:]
    
    return feature_vector.tolist()

# 根据特征选择策略标签
def select_strategy_label(features):
    """根据特征向量选择合适的策略标签"""
    # 从特征中提取任务复杂度、紧急程度和资源可用性
    task_complexity = features[0]
    task_urgency = features[1]
    resource_availability = features[2]
    team_size = features[3]
    
    # 基于规则的策略选择
    if task_urgency > 0.7 or resource_availability < 0.3:
        # 紧急任务或资源有限时，使用sequential策略
        return 0  # sequential
    elif task_complexity > 0.6 and team_size > 10:
        # 复杂任务且团队规模大时，使用hierarchical策略
        return 2  # hierarchical
    elif task_complexity < 0.4 and resource_availability > 0.5:
        # 简单任务且资源充足时，使用parallel策略
        return 1  # parallel
    else:
        # 其他情况使用adaptive策略
        return 3  # adaptive

# 根据特征选择情感标签
def select_emotion_label(features):
    """根据特征向量选择合适的情感标签"""
    # 从特征中提取任务复杂度、紧急程度和资源可用性
    task_complexity = features[0]
    task_urgency = features[1]
    resource_availability = features[2]
    
    # 基于规则的情感选择
    if resource_availability > 0.7 and task_urgency < 0.3:
        # 资源充足且不紧急时，心情愉快
        return 0  # happy
    elif task_complexity > 0.8 and resource_availability < 0.4:
        # 复杂任务且资源不足时，可能会生气
        return 2  # angry
    elif task_urgency > 0.8:
        # 非常紧急时，可能会感到害怕
        return 3  # fear
    elif task_complexity < 0.3 and task_urgency < 0.3:
        # 简单且不紧急时，心情平静
        return 6  # neutral
    elif random.random() > 0.8:
        # 随机选择一个惊喜的情感
        return 4  # surprise
    elif task_complexity > 0.7:
        # 复杂任务可能会带来一些负面情绪
        return 1  # sad
    else:
        # 默认使用中性情感
        return 6  # neutral

# 生成策略的one-hot编码
def get_strategy_one_hot(label):
    """将策略标签转换为one-hot编码"""
    one_hot = [0] * len(strategy_mapping)
    one_hot[label] = 1
    return one_hot

# 生成情感的one-hot编码
def get_emotion_one_hot(label):
    """将情感标签转换为one-hot编码"""
    one_hot = [0] * len(emotion_mapping)
    one_hot[label] = 1
    return one_hot

# 生成样本数据
def generate_sample():
    """生成单个样本数据"""
    # 生成特征向量
    features = generate_feature_vector()
    
    # 选择策略和情感标签
    strategy_label = select_strategy_label(features)
    emotion_label = select_emotion_label(features)
    
    # 生成one-hot编码
    strategy_one_hot = get_strategy_one_hot(strategy_label)
    emotion_one_hot = get_emotion_one_hot(emotion_label)
    
    # 合并标签
    combined_labels = strategy_one_hot + emotion_one_hot
    
    # 构建样本字典
    sample = {
        'features': features,
        'strategy_label': strategy_label,
        'strategy_name': strategy_mapping[strategy_label],
        'strategy_one_hot': strategy_one_hot,
        'emotion_label': emotion_label,
        'emotion_name': emotion_mapping[emotion_label],
        'emotion_one_hot': emotion_one_hot,
        'combined_labels': combined_labels
    }
    
    return sample

# 生成并保存数据
def generate_and_save_data(num_samples=1000, split_ratio=[0.7, 0.15, 0.15]):
    """生成数据并按比例分割保存为训练集、验证集和测试集"""
    # 确保数据目录存在
    os.makedirs(DATA_DIR, exist_ok=True)
    
    # 生成样本数据
    print(f"Generating {num_samples} samples...")
    samples = [generate_sample() for _ in range(num_samples)]
    
    # 分割数据
    train_size = int(num_samples * split_ratio[0])
    val_size = int(num_samples * split_ratio[1])
    
    train_data = samples[:train_size]
    val_data = samples[train_size:train_size+val_size]
    test_data = samples[train_size+val_size:]
    
    # 保存数据
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # 保存为JSON文件
    train_path = os.path.join(DATA_DIR, 'train_data.json')
    val_path = os.path.join(DATA_DIR, 'val_data.json')
    test_path = os.path.join(DATA_DIR, 'test_data.json')
    
    with open(train_path, 'w', encoding='utf-8') as f:
        json.dump(train_data, f, indent=2, ensure_ascii=False)
    
    with open(val_path, 'w', encoding='utf-8') as f:
        json.dump(val_data, f, indent=2, ensure_ascii=False)
    
    with open(test_path, 'w', encoding='utf-8') as f:
        json.dump(test_data, f, indent=2, ensure_ascii=False)
    
    print(f"Data generation complete!")
    print(f"Training data: {len(train_data)} samples saved to {train_path}")
    print(f"Validation data: {len(val_data)} samples saved to {val_path}")
    print(f"Test data: {len(test_data)} samples saved to {test_path}")

# 生成额外的任务数据，用于ManagementTaskDataset
def generate_task_data(num_samples=1000):
    """生成ManagementTaskDataset所需的数据格式"""
    task_data = []
    
    for i in range(num_samples):
        # 生成特征向量
        features = generate_feature_vector()
        
        # 选择策略和情感标签
        strategy_label = select_strategy_label(features)
        emotion_label = select_emotion_label(features)
        
        # 构建任务数据字典
        task = {
            'id': f'task_{i}',
            'features': features,
            'strategy_label': strategy_label,
            'emotion_label': emotion_label,
            'strategy_name': strategy_mapping[strategy_label],
            'emotion_name': emotion_mapping[emotion_label],
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'source': 'generated_sample_data'
            }
        }
        
        task_data.append(task)
    
    # 保存数据
    task_data_path = os.path.join(DATA_DIR, 'management_tasks.json')
    with open(task_data_path, 'w', encoding='utf-8') as f:
        json.dump(task_data, f, indent=2, ensure_ascii=False)
    
    print(f"Task data: {len(task_data)} samples saved to {task_data_path}")

# 主函数
if __name__ == "__main__":
    print("Generating sample data for A_management model...")
    
    # 生成标准数据格式
    generate_and_save_data(num_samples=1000)
    
    # 生成ManagementTaskDataset所需的数据格式
    generate_task_data(num_samples=1000)
    
    print("All sample data generated successfully!")