# 增强的统一语言模型训练功能文档

## 功能概述

本模块为统一语言模型(`UnifiedLanguageModel`)添加了全面的训练功能增强，包括：

1. **真实数据管道**：支持从文件系统加载多语言训练数据
2. **持续学习/增量学习**：能够基于低置信度样本进行持续优化
3. **知识迁移**：支持在不同语言之间进行知识迁移学习
4. **自我评估**：提供模型性能评估和反馈机制
5. **TensorBoard集成**：支持训练过程可视化

## 数据结构与格式

### 数据目录结构

训练数据应按照以下目录结构组织：

```
data/
├── zh/  # 中文数据
│   └── sample_training_data.json
├── en/  # 英文数据
│   └── sample_training_data.json
├── de/  # 德文数据
│   └── sample_training_data.json
├── ja/  # 日文数据
│   └── sample_training_data.json
└── ...  # 其他语言数据
```

### 数据文件格式

每个语言的数据文件应为JSON格式，包含以下字段：

```json
[
  {"text": "文本内容", "label": "情感标签", "language": "语言代码", "intensity": 情感强度},
  ...
]
```

支持的情感标签：`anger`, `disgust`, `fear`, `joy`, `neutral`, `sadness`, `surprise`

## 主要功能接口

### 1. 基本训练功能

```python
from unified_language_model import UnifiedLanguageModel

# 初始化模型
model = UnifiedLanguageModel(mode="enhanced")

# 执行基本训练
result = model.train_model(
    data_path="./data",          # 数据路径
    languages=["zh", "en"],       # 要训练的语言
    epochs=10,                    # 训练轮数
    batch_size=32,                # 批次大小
    learning_rate=0.0001,         # 学习率
    use_tensorboard=True          # 是否使用TensorBoard
)
```

### 2. 增量学习功能

增量学习允许模型基于之前预测的低置信度样本进行持续优化：

```python
# 执行增量学习
result = model.incremental_train(
    data_path="./data",          # 数据路径
    languages=["en"],             # 要训练的语言
    epochs=5,                     # 训练轮数
    batch_size=16,                # 批次大小
    learning_rate=0.00001,        # 较小的学习率以避免过拟合
    confidence_threshold=0.7      # 低置信度阈值
)

# 查看低置信度样本
low_confidence_samples = model.get_low_confidence_samples()
```

### 3. 知识迁移功能

知识迁移允许模型将从一种语言学习的知识迁移到另一种语言：

```python
# 执行知识迁移学习
result = model.transfer_learn(
    source_language="en",        # 源语言（已训练好的语言）
    target_language=["de"],       # 目标语言（要迁移学习的语言）
    data_path="./data",          # 数据路径
    epochs=5,                     # 训练轮数
    batch_size=16,                # 批次大小
    learning_rate=0.00001,        # 较小的学习率
    transfer_strength=0.5         # 迁移强度（0-1之间的值）
)
```

### 4. 模型评估功能

```python
# 评估模型性能
metrics = model.evaluate_model(
    data_path="./data",          # 评估数据路径
    languages=["zh", "en", "de"] # 要评估的语言
)

print(f"模型准确率: {metrics['accuracy']}")
print(f"F1分数: {metrics['f1_score']}")
print(f"情感强度MSE: {metrics['intensity_mse']}")
```

### 5. 模型保存与加载

```python
# 保存模型
model.save_model("./models/my_trained_model")

# 加载模型
new_model = UnifiedLanguageModel(mode="enhanced")
new_model.load_model("./models/my_trained_model")
```

## API接口使用

### 1. 训练接口

```bash
curl -X POST http://localhost:5002/train \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "B_language",
    "epochs": 10,
    "batch_size": 32,
    "learning_rate": 0.0001,
    "data_path": "./data",
    "languages": ["zh", "en", "de", "ja"]
  }'
```

### 2. 增量学习接口

```bash
curl -X POST http://localhost:5002/train \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "B_language",
    "epochs": 5,
    "batch_size": 16,
    "learning_rate": 0.00001,
    "data_path": "./data",
    "languages": ["fr"],
    "use_incremental": true
  }'
```

### 3. 知识迁移接口

```bash
curl -X POST http://localhost:5002/train \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "B_language",
    "epochs": 5,
    "batch_size": 16,
    "learning_rate": 0.00001,
    "data_path": "./data",
    "languages": ["ru"],
    "use_transfer": true,
    "source_language": "en"
  }'
```

## 测试脚本使用

本模块提供了一个测试脚本来验证所有增强功能：

```bash
# 运行测试脚本
python test_enhanced_training.py
```

测试脚本会验证以下功能：
1. 模型初始化
2. 基本训练功能
3. 增量学习功能
4. 知识迁移功能
5. 模型评估功能
6. 模型保存和加载功能
7. API接口功能（可选）

## 配置参数

### 模型配置

在`UnifiedLanguageModel`初始化时，可以指定以下配置参数：

```python
model = UnifiedLanguageModel(
    mode="enhanced",                # 模型模式：base或enhanced
    base_model="xlm-roberta-base",  # 基础预训练模型
    emotion_categories=["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"],
    supported_languages=["zh", "en", "de", "ja", "ru"],
    use_tensorboard=True,           # 是否使用TensorBoard
    log_dir="./logs"                # TensorBoard日志目录
)
```

### 训练配置

训练相关配置可以在`config/training_config.json`文件中设置：

```json
{
  "epochs": 100,
  "batch_size": 32,
  "learning_rate": 0.0001,
  "optimizer": "adamw",
  "weight_decay": 0.01,
  "loss_function": "cross_entropy",
  "lr_scheduler": "cosine",
  "early_stopping_patience": 10,
  "gradient_clipping": 1.0
}
```

## 注意事项

1. 增量学习和知识迁移功能需要先进行基础训练
2. 训练大型模型时建议使用GPU加速
3. 数据量不足时，增量学习和知识迁移可能效果不佳
4. TensorBoard日志可以通过`tensorboard --logdir=./logs`命令查看
5. 低置信度样本存储在内存中，大量样本可能会占用较多内存

## 版本信息

- 增强版发布日期：2024-09-16
- 支持的模型：xlm-roberta-base, xlm-roberta-large
- 支持的语言：中文、英文、德文、日文、俄文等
- 支持的情感类别：7种基础情感类别

## 问题反馈

如有任何问题或建议，请联系系统管理员或开发团队。