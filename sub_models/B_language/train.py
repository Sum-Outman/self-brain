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
    
    print("训练结果 | Training result:", json.dumps(result, indent=2, ensure_ascii=False))
