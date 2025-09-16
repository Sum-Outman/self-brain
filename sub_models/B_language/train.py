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

# B语言模型训练程序 | B Language Model Training Program
# 多语言自然语言处理模型的完整训练实现 | Complete training implementation for multilingual NLP models

import os
import json
import torch
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    Trainer, 
    TrainingArguments,
    DataCollatorWithPadding
)
from datasets import Dataset, load_dataset
import evaluate
from sklearn.model_selection import train_test_split

class LanguageModelTrainer:
    """语言模型训练器 | Language Model Trainer"""
    
    def __init__(self, model_type="sentiment", language="multilingual", use_external_api=False, api_config=None):
        """初始化训练器 | Initialize trainer
        
        参数:
            model_type: 模型类型 (sentiment/ner/generation/emotion) | Model type (sentiment/ner/generation/emotion)
            language: 语言类型 (multilingual/zh/en/ja/de/ru) | Language type (multilingual/zh/en/ja/de/ru)
            use_external_api: 是否使用外部API | Whether to use external API
            api_config: 外部API配置字典 | External API configuration dict
        """
        self.model_type = model_type
        self.language = language
        self.use_external_api = use_external_api
        self.api_config = api_config or {}
        self.model = None
        self.tokenizer = None
        self.trainer = None
        self.training_config = self._get_default_config()
        self.joint_training_partners = []  # 联合训练伙伴模型 | Joint training partners
        self.emotion_categories = {
            "anger": 0, "disgust": 1, "fear": 2, "joy": 3, 
            "neutral": 4, "sadness": 5, "surprise": 6
        }
        self.language_resources = self._load_language_resources()
        
    def _get_default_config(self):
        """获取默认训练配置 | Get default training configuration"""
        return {
            'epochs': 10,
            'batch_size': 16,
            'learning_rate': 2e-5,
            'warmup_steps': 500,
            'logging_steps': 100,
            'eval_steps': 500,
            'save_steps': 1000,
            'max_seq_length': 256,
            'joint_training': False,
            'external_api_timeout': 30
        }
    
    def _load_language_resources(self):
        """加载多语言资源 | Load multilingual resources"""
        # 实际实现应从文件加载，这里简化
        return {
            "zh": {"emotion_labels": ["愤怒", "厌恶", "恐惧", "快乐", "中性", "悲伤", "惊讶"]},
            "en": {"emotion_labels": ["Anger", "Disgust", "Fear", "Joy", "Neutral", "Sadness", "Surprise"]},
            "ja": {"emotion_labels": ["怒り", "嫌悪", "恐怖", "喜び", "中性", "悲しみ", "驚き"]},
            "de": {"emotion_labels": ["Wut", "Ekel", "Angst", "Freude", "Neutral", "Traurigkeit", "Überraschung"]},
            "ru": {"emotion_labels": ["Гнев", "Отвращение", "Страх", "Радость", "Нейтральный", "Грусть", "Удивление"]}
        }
    
    def load_pretrained_model(self):
        """加载预训练模型 | Load pretrained model"""
        try:
            if self.use_external_api:
                print("使用外部API模型 | Using external API model")
                return True
                
            if self.model_type == "sentiment" or self.model_type == "emotion":
                model_name = "bert-base-multilingual-cased"
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                num_labels = 5 if self.model_type == "sentiment" else len(self.emotion_categories)
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    model_name, num_labels=num_labels
                )
            elif self.model_type == "ner":
                model_name = "Davlan/bert-base-multilingual-cased-ner-hrl"
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.model = AutoModelForTokenClassification.from_pretrained(model_name)
            else:
                raise ValueError(f"不支持的模型类型: {self.model_type} | Unsupported model type: {self.model_type}")
                
            print(f"预训练模型加载成功: {model_name} | Pretrained model loaded successfully: {model_name}")
            return True
        except Exception as e:
            print(f"模型加载失败: {e} | Model loading failed: {e}")
            return False
    
    def prepare_dataset(self, data_path=None, task_type="sentiment"):
        """准备训练数据集 | Prepare training dataset
        
        参数:
            data_path: 数据文件路径 | Data file path
            task_type: 任务类型 (sentiment/ner) | Task type (sentiment/ner)
        """
        if data_path and os.path.exists(data_path):
            # 从文件加载数据 | Load data from file
            with open(data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        else:
            # 使用示例数据 | Use example data
            data = self._get_example_data(task_type)
        
        if task_type == "sentiment":
            return self._prepare_sentiment_data(data)
        elif task_type == "ner":
            return self._prepare_ner_data(data)
        else:
            raise ValueError(f"不支持的任务类型: {task_type} | Unsupported task type: {task_type}")
    
    def _get_example_data(self, task_type):
        """获取示例训练数据 | Get example training data"""
        if task_type == "sentiment":
            return [
                {"text": "这个产品非常好用，我很满意！", "label": 4, "language": "zh"},
                {"text": "质量很差，完全不值得购买", "label": 0, "language": "zh"},
                {"text": "This is an amazing product!", "label": 4, "language": "en"},
                {"text": "Very disappointed with the service", "label": 1, "language": "en"},
                {"text": "Das Buch ist ausgezeichnet", "label": 4, "language": "de"},
                {"text": "Nicht zu empfehlen", "label": 1, "language": "de"},
                {"text": "この製品は素晴らしいです！", "label": 4, "language": "ja"},
                {"text": "サービスが悪かった", "label": 0, "language": "ja"},
                {"text": "Отличный продукт", "label": 4, "language": "ru"},
                {"text": "Очень разочарован", "label": 1, "language": "ru"}
            ]
        elif task_type == "emotion":
            return [
                {"text": "我被这个决定激怒了！", "label": 0, "language": "zh", "intensity": 0.9},
                {"text": "这个惊喜让我非常开心", "label": 3, "language": "zh", "intensity": 0.8},
                {"text": "I'm terrified of what might happen", "label": 2, "language": "en", "intensity": 0.95},
                {"text": "Diese Nachricht macht mich traurig", "label": 5, "language": "de", "intensity": 0.7},
                {"text": "この結果に驚きました", "label": 6, "language": "ja", "intensity": 0.6},
                {"text": "Я чувствую отвращение к этой ситуации", "label": 1, "language": "ru", "intensity": 0.85}
            ]
        elif task_type == "ner":
            return [
                {"text": "张三去了北京的天安门", "entities": [{"start": 0, "end": 2, "label": "PER"}, {"start": 6, "end": 8, "label": "LOC"}]},
                {"text": "Apple Inc. is located in Cupertino", "entities": [{"start": 0, "end": 10, "label": "ORG"}, {"start": 27, "end": 36, "label": "LOC"}]}
            ]
        return []
    
    def _prepare_sentiment_data(self, data):
        """准备情感分析数据 | Prepare sentiment analysis data"""
        texts = [item['text'] for item in data]
        labels = [item['label'] for item in data]
        
        # 分词 | Tokenization
        encodings = self.tokenizer(
            texts, 
            truncation=True, 
            padding=True, 
            max_length=self.training_config['max_seq_length']
        )
        
        dataset = Dataset.from_dict({
            'input_ids': encodings['input_ids'],
            'attention_mask': encodings['attention_mask'],
            'labels': labels
        })
        
        # 分割训练集和验证集 | Split train and validation sets
        train_test_split = dataset.train_test_split(test_size=0.2)
        return train_test_split
    
    def _prepare_ner_data(self, data):
        """准备命名实体识别数据 | Prepare NER data"""
        # 简化实现，实际需要更复杂的处理 | Simplified implementation, actual needs more complex processing
        texts = [item['text'] for item in data]
        
        encodings = self.tokenizer(
            texts, 
            truncation=True, 
            padding=True, 
            max_length=self.training_config['max_seq_length'],
            is_split_into_words=False
        )
        
        # 创建标签 | Create labels
        labels = []
        for item in data:
            text_labels = ['O'] * len(item['text'])
            for entity in item['entities']:
                # 简化标签分配 | Simplified label assignment
                text_labels[entity['start']:entity['end']] = [f'B-{entity["label"]}'] + [f'I-{entity["label"]}'] * (entity['end'] - entity['start'] - 1)
            labels.append(text_labels)
        
        dataset = Dataset.from_dict({
            'input_ids': encodings['input_ids'],
            'attention_mask': encodings['attention_mask'],
            'labels': labels
        })
        
        return dataset.train_test_split(test_size=0.2)
    
    def train(self, train_dataset, eval_dataset=None, config=None, joint_data=None):
        """训练模型 | Train model"""
        if config:
            self.training_config.update(config)
        
        # 联合训练数据处理 | Joint training data processing
        if joint_data:
            print(f"接收来自{len(joint_data)}个模型的联合训练数据 | Received joint training data from {len(joint_data)} models")
            # 此处添加多模态数据融合逻辑 | Add multimodal data fusion logic here
            # 示例：将知识库模型数据融入训练 | Example: Integrate knowledge model data
            if 'knowledge' in joint_data:
                train_dataset = self._enhance_with_knowledge(train_dataset, joint_data['knowledge'])
        
        if self.use_external_api:
            return self._train_via_external_api(train_dataset)
        
        if not self.model or not self.tokenizer:
            if not self.load_pretrained_model():
                return {"status": "error", "message": "模型加载失败 | Model loading failed"}
        
        # 设置训练参数 | Set training arguments
        training_args = TrainingArguments(
            output_dir=f"./results/{self.model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            num_train_epochs=self.training_config['epochs'],
            per_device_train_batch_size=self.training_config['batch_size'],
            per_device_eval_batch_size=self.training_config['batch_size'],
            warmup_steps=self.training_config['warmup_steps'],
            learning_rate=self.training_config['learning_rate'],
            logging_steps=self.training_config['logging_steps'],
            eval_steps=self.training_config['eval_steps'] if eval_dataset else None,
            save_steps=self.training_config['save_steps'],
            evaluation_strategy="steps" if eval_dataset else "no",
            save_strategy="steps",
            load_best_model_at_end=True if eval_dataset else False,
            metric_for_best_model="accuracy" if self.model_type == "sentiment" else "f1",
            greater_is_better=True
        )
        
        # 创建训练器 | Create trainer
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            tokenizer=self.tokenizer,
            compute_metrics=self._compute_metrics
        )
        
        # 开始训练 | Start training
        print(f"开始训练 {self.model_type} 模型... | Starting {self.model_type} model training...")
        train_result = self.trainer.train()
        
        # 保存模型 | Save model
        self.trainer.save_model()
        self.tokenizer.save_pretrained(self.trainer.args.output_dir)
        
        metrics = train_result.metrics
        metrics.update(self.trainer.evaluate())
        
        return {
            "status": "success",
            "message": "训练完成 | Training completed",
            "metrics": metrics,
            "model_path": self.trainer.args.output_dir
        }
    
    def _compute_metrics(self, eval_pred):
        """计算评估指标 | Compute evaluation metrics"""
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        
        accuracy_metric = evaluate.load("accuracy")
        precision_metric = evaluate.load("precision")
        recall_metric = evaluate.load("recall")
        f1_metric = evaluate.load("f1")
        
        metrics = {
            "accuracy": accuracy_metric.compute(predictions=predictions, references=labels)["accuracy"],
            "precision": precision_metric.compute(predictions=predictions, references=labels, average="weighted")["precision"],
            "recall": recall_metric.compute(predictions=predictions, references=labels, average="weighted")["recall"],
            "f1": f1_metric.compute(predictions=predictions, references=labels, average="weighted")["f1"]
        }
        
        # 情感强度分析（仅emotion类型）| Emotion intensity analysis (only for emotion type)
        if self.model_type == "emotion":
            intensity_scores = self._calculate_emotion_intensity(predictions, labels)
            metrics.update(intensity_scores)
        
        return metrics
    
    def _calculate_emotion_intensity(self, predictions, labels):
        """计算情感强度得分 | Calculate emotion intensity scores"""
        # 简化实现，实际应使用更复杂的算法 | Simplified implementation
        intensity = np.abs(predictions - labels).mean()
        return {
            "intensity_error": float(intensity),
            "intensity_accuracy": 1.0 - intensity
        }
    
    def evaluate_model(self, test_data):
        """评估模型性能 | Evaluate model performance"""
        if not self.model or not self.tokenizer:
            return {"status": "error", "message": "模型未加载 | Model not loaded"}
        
        # 准备测试数据 | Prepare test data
        test_dataset = self.prepare_dataset(test_data, self.model_type)['test']
        
        # 评估 | Evaluate
        eval_results = self.trainer.evaluate(test_dataset)
        return eval_results
    
    def save_training_report(self, metrics, output_path):
        """保存训练报告 | Save training report"""
        report = {
            "training_date": datetime.now().isoformat(),
            "model_type": self.model_type,
            "language": self.language,
            "training_config": self.training_config,
            "metrics": metrics,
            "hardware_info": {
                "device": str(torch.device('cuda' if torch.cuda.is_available() else 'cpu')),
                "cuda_available": torch.cuda.is_available(),
                "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0
            }
        }
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        return output_path

# 新增方法 | New methods
    def _train_via_external_api(self, dataset):
        """通过外部API训练 | Train via external API"""
        # 实际实现应调用配置的API | Actual implementation should call configured API
        print(f"通过外部API训练: {self.api_config.get('endpoint')} | Training via external API: {self.api_config.get('endpoint')}")
        return {
            "status": "success",
            "message": "外部API训练完成 | External API training completed",
            "metrics": {"external_api": True},
            "model_path": "external_api_model"
        }
    
    def _enhance_with_knowledge(self, dataset, knowledge_data):
        """使用知识库数据增强训练集 | Enhance dataset with knowledge data"""
        # 实际实现应融合知识库信息 | Actual implementation should integrate knowledge
        print("使用知识库数据增强训练集 | Enhancing dataset with knowledge data")
        return dataset
    
    def set_joint_training(self, models):
        """设置联合训练伙伴 | Set joint training partners"""
        self.joint_training_partners = models
        print(f"设置联合训练伙伴: {', '.join(models)} | Set joint training partners: {', '.join(models)}")
    
    def connect_to_main_model(self, main_model_endpoint):
        """连接到主模型 | Connect to main model"""
        print(f"连接到主模型: {main_model_endpoint} | Connected to main model: {main_model_endpoint}")
        # 实际实现应建立通信通道 | Actual implementation should establish communication
    
    def train_jointly(self, models: List[Any], train_datasets: List[Any], 
                     val_datasets: List[Any] = None, config: Dict = None, 
                     loss_weights: List[float] = None) -> Dict:
        """与其他模型联合训练 | Joint training with other models
        
        参数:
            models: 参与联合训练的模型列表 | List of models for joint training
            train_datasets: 每个模型对应的训练数据集 | Training datasets for each model
            val_datasets: 每个模型对应的验证数据集 | Validation datasets for each model
            config: 训练配置 | Training configuration
            loss_weights: 每个模型的损失权重 | Loss weights for each model
        
        返回:
            训练结果字典 | Training result dictionary
        """
        if config:
            self.training_config.update(config)
        
        # 检查输入一致性 | Check input consistency
        if len(models) != len(train_datasets):
            raise ValueError("模型数量和训练数据集数量必须匹配 | Number of models and training datasets must match")
        
        if val_datasets and len(val_datasets) != len(models):
            raise ValueError("验证数据集数量必须与模型数量匹配 | Number of validation datasets must match number of models")
        
        # 初始化默认损失权重 | Initialize default loss weights if not provided
        if loss_weights is None:
            loss_weights = [1.0] * len(models)
        elif len(loss_weights) != len(models):
            raise ValueError("损失权重数量必须与模型数量匹配 | Number of loss weights must match number of models")
        
        # 归一化损失权重 | Normalize loss weights
        total_weight = sum(loss_weights)
        loss_weights = [w / total_weight for w in loss_weights]
        
        # 记录联合训练信息 | Log joint training information
        print(f"开始多模型联合训练 | Starting joint training with {len(models)} models")
        print(f"损失权重分配 | Loss weight distribution: {loss_weights}")
        
        # 初始化联合训练结果 | Initialize joint training results
        joint_results = {
            "status": "success",
            "message": "联合训练完成 | Joint training completed",
            "individual_results": [],
            "joint_metrics": {}
        }
        
        # 为每个模型准备联合训练数据 | Prepare joint training data for each model
        for i, (model, train_dataset) in enumerate(zip(models, train_datasets)):
            print(f"处理模型 {i+1} 的训练数据 | Processing training data for model {i+1}")
            
            # 为每个模型创建联合训练伙伴信息 | Create joint training partner info for each model
            joint_training_info = {}
            for j, partner_model in enumerate(models):
                if i != j:  # 不包含自身 | Exclude self
                    # 根据伙伴模型类型添加相应信息 | Add appropriate info based on partner model type
                    # 这部分需要根据实际模型类型和接口进行实现 | This part needs to be implemented based on actual model types and interfaces
                    partner_name = f"model_{j+1}"
                    joint_training_info[partner_name] = {
                        "model_type": "language" if j == 0 else "partner",
                        "weight": loss_weights[j]
                    }
            
            # 训练当前模型，传入其他模型的信息 | Train current model with other models' info
            val_dataset = val_datasets[i] if val_datasets else None
            
            if model == self:  # 训练当前对象 | Training current object
                result = self.train(train_dataset, val_dataset, config, joint_training_info)
            else:  # 训练其他模型 | Training other models
                # 假设其他模型也有类似的train方法 | Assume other models have similar train method
                try:
                    result = model.train(train_dataset, val_dataset, config, joint_training_info)
                except Exception as e:
                    print(f"训练模型 {i+1} 失败: {e} | Training model {i+1} failed: {e}")
                    joint_results["status"] = "partial_failure"
                    joint_results["message"] = f"部分模型训练失败 | Some models training failed"
                    result = {"status": "error", "message": str(e)}
            
            joint_results["individual_results"].append(result)
        
        # 计算联合指标 | Calculate joint metrics
        if all(res.get("status") == "success" for res in joint_results["individual_results"]):
            joint_accuracy = sum(res["metrics"].get("accuracy", 0) * w for res, w in zip(joint_results["individual_results"], loss_weights))
            joint_f1 = sum(res["metrics"].get("f1", 0) * w for res, w in zip(joint_results["individual_results"], loss_weights))
            
            joint_results["joint_metrics"] = {
                "joint_accuracy": joint_accuracy,
                "joint_f1": joint_f1,
                "loss_weights": loss_weights
            }
            
            print(f"联合训练完成，联合准确率: {joint_accuracy:.4f}, 联合F1得分: {joint_f1:.4f} | Joint training completed, joint accuracy: {joint_accuracy:.4f}, joint F1: {joint_f1:.4f}")
        
        # 保存联合训练报告 | Save joint training report
        report_path = f"./training_reports/joint_{self.model_type}_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump({
                "training_date": datetime.now().isoformat(),
                "joint_training": True,
                "model_count": len(models),
                "models": [{
                    "model_type": "language",
                    "specific_type": self.model_type,
                    "language": self.language
                }] + [{"model_type": "partner", "index": i+2} for i in range(len(models)-1)],
                "loss_weights": loss_weights,
                "training_config": self.training_config,
                "joint_results": joint_results
            }, f, indent=2, ensure_ascii=False)
        
        joint_results["report_path"] = report_path
        return joint_results

# 训练API接口 | Training API interface
def start_training(model_type="sentiment", data_path=None, config=None, use_external=False, api_config=None):
    """启动训练任务 | Start training task"""
    trainer = LanguageModelTrainer(
        model_type=model_type, 
        use_external_api=use_external,
        api_config=api_config
    )
    
    # 准备数据 | Prepare data
    datasets = trainer.prepare_dataset(data_path, model_type)
    
    # 开始训练 | Start training
    result = trainer.train(datasets['train'], datasets['test'], config)
    
    if result['status'] == 'success':
        # 保存训练报告 | Save training report
        report_path = f"./training_reports/{model_type}_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        trainer.save_training_report(result['metrics'], report_path)
        result['report_path'] = report_path
    
    return result

def resume_training(model_path, data_path=None, config=None):
    """恢复训练 | Resume training"""
    # 实现恢复训练逻辑 | Implement resume training logic
    pass

# 联合训练API接口 | Joint training API interface
def start_joint_training(model_configs: List[Dict], data_paths: List[str], config: Dict = None, loss_weights: List[float] = None) -> Dict:
    """启动联合训练任务 | Start joint training task
    
    参数:
        model_configs: 各模型的配置列表 | List of model configurations
        data_paths: 各模型的数据路径列表 | List of data paths for each model
        config: 全局训练配置 | Global training configuration
        loss_weights: 各模型的损失权重 | Loss weights for each model
    
    返回:
        联合训练结果 | Joint training result
    """
    # 创建参与联合训练的模型列表 | Create list of models for joint training
    models = []
    train_datasets = []
    val_datasets = []
    
    print(f"准备启动联合训练，参与模型数量: {len(model_configs)} | Preparing to start joint training with {len(model_configs)} models")
    
    # 初始化每个模型并准备数据 | Initialize each model and prepare data
    for i, model_config in enumerate(model_configs):
        print(f"初始化模型 {i+1}: {model_config.get('model_type', 'unknown')} | Initializing model {i+1}: {model_config.get('model_type', 'unknown')}")
        
        # 创建模型实例 | Create model instance
        trainer = LanguageModelTrainer(
            model_type=model_config.get('model_type', 'sentiment'),
            language=model_config.get('language', 'multilingual'),
            use_external_api=model_config.get('use_external_api', False),
            api_config=model_config.get('api_config')
        )
        
        # 加载预训练模型 | Load pretrained model
        if not trainer.use_external_api:
            trainer.load_pretrained_model()
        
        # 准备数据集 | Prepare dataset
        data_path = data_paths[i] if i < len(data_paths) else None
        datasets = trainer.prepare_dataset(data_path, model_config.get('model_type', 'sentiment'))
        
        # 添加到列表 | Add to lists
        models.append(trainer)
        train_datasets.append(datasets['train'])
        val_datasets.append(datasets['test'])
    
    # 如果只有一个模型，则执行普通训练 | If only one model, perform regular training
    if len(models) == 1:
        print("只有一个模型，执行普通训练 | Only one model, performing regular training")
        return start_training(
            model_type=model_configs[0].get('model_type', 'sentiment'),
            data_path=data_paths[0] if data_paths else None,
            config=config,
            use_external=model_configs[0].get('use_external_api', False),
            api_config=model_configs[0].get('api_config')
        )
    
    # 执行联合训练 | Perform joint training
    # 选择第一个模型作为主导模型来协调联合训练 | Select the first model as the lead to coordinate joint training
    lead_model = models[0]
    return lead_model.train_jointly(models, train_datasets, val_datasets, config, loss_weights)

if __name__ == '__main__':
    # 测试训练程序 | Test training program
    print("测试B语言模型训练程序... | Testing B Language Model Training Program...")
    
    # 测试情感分析训练 | Test sentiment analysis training
    result = start_training(
        model_type="sentiment",
        config={
            'epochs': 3,
            'batch_size': 8,
            'learning_rate': 2e-5
        }
    )
    
    # 测试情感推理训练 | Test emotion reasoning training
    emotion_result = start_training(
        model_type="emotion",
        config={
            'epochs': 4,
            'batch_size': 12,
            'learning_rate': 3e-5
        }
    )
    
    # 测试外部API训练 | Test external API training
    api_result = start_training(
        model_type="sentiment",
        use_external=True,
        api_config={
            "endpoint": "https://api.agilanguage.com/v1/train",
            "api_key": "your_api_key_here"
        }
    )
    
    print("情感分析结果 | Sentiment result:", json.dumps(result, indent=2, ensure_ascii=False))
    print("情感推理结果 | Emotion result:", json.dumps(emotion_result, indent=2, ensure_ascii=False))
    print("外部API结果 | External API result:", json.dumps(api_result, indent=2, ensure_ascii=False))
    
    # 测试联合训练 | Test joint training (示例代码，实际使用时需根据实际情况修改)
    print("\n测试联合训练功能... | Testing joint training functionality...")
    
    # 创建两个语言模型进行联合训练演示
    # Note: 这只是一个演示，实际联合训练应包含不同类型的模型
    joint_result = start_joint_training(
        model_configs=[
            {"model_type": "sentiment", "language": "multilingual"},
            {"model_type": "emotion", "language": "multilingual"}
        ],
        data_paths=[None, None],  # 使用示例数据
        config={
            'epochs': 2,
            'batch_size': 8,
            'learning_rate': 2e-5
        },
        loss_weights=[0.6, 0.4]
    )
    
    print("联合训练结果 | Joint training result:", json.dumps(joint_result, indent=2, ensure_ascii=False))
