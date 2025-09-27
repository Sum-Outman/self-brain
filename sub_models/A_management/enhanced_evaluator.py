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
增强的评估器，用于评估管理模型的情感分析和策略预测性能
"""

import torch
import numpy as np
import json
import os
import logging
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from tqdm import tqdm
import seaborn as sns
from datetime import datetime

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('A_management_evaluator')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ModelEvaluator:
    """
    模型评估器类，用于评估管理模型的性能
    """
    def __init__(self, model, config=None):
        """
        初始化评估器
        
        参数:
            model: 要评估的模型
            config: 评估配置
        """
        self.model = model.to(device)
        self.model.eval()  # 设置为评估模式
        
        # 默认配置
        default_config = {
            'batch_size': 32,
            'metrics_dir': './metrics',
            'figures_dir': './figures',
            'output_dir': './evaluation_results',
            'emotion_labels': ['neutral', 'joy', 'sadness', 'anger', 'fear', 'surprise', 'disgust'],
            'strategy_labels': ['strategy_0', 'strategy_1', 'strategy_2', 'strategy_3'],
            'save_plots': True,
            'save_metrics': True,
            'verbose': True
        }
        
        # 合并配置
        self.config = {**default_config, **(config or {})}
        
        # 创建输出目录
        os.makedirs(self.config['output_dir'], exist_ok=True)
        os.makedirs(self.config['metrics_dir'], exist_ok=True)
        os.makedirs(self.config['figures_dir'], exist_ok=True)
        
        # 情感和策略标签
        self.emotion_labels = self.config['emotion_labels']
        self.strategy_labels = self.config['strategy_labels']
        
    def evaluate(self, data_loader, save_results=True, prefix='eval'):
        """
        评估模型性能
        
        参数:
            data_loader: 数据加载器
            save_results: 是否保存评估结果
            prefix: 结果文件的前缀
        
        返回:
            评估指标字典
        """
        all_strategy_preds = []
        all_strategy_labels = []
        all_emotion_preds = []
        all_emotion_labels = []
        all_strategy_probs = []
        all_emotion_probs = []
        
        # 用于记录下属模型的评估结果
        sub_model_evaluations = {}
        
        with torch.no_grad():
            progress_bar = tqdm(data_loader, desc="Evaluation", leave=False)
            
            for batch in progress_bar:
                features = batch['features']
                strategy_labels = batch['strategy_labels']
                emotion_labels = batch['emotion_labels']
                sub_model_outputs = batch['sub_model_outputs']
                
                # 批量处理每个样本
                batch_strategy_preds = []
                batch_emotion_preds = []
                batch_strategy_probs = []
                batch_emotion_probs = []
                
                for i in range(len(features)):
                    # 前向传播
                    strategy_probs, emotion_probs = self.model(features[i], sub_model_outputs[i])
                    
                    # 获取预测结果
                    _, strategy_pred = torch.max(strategy_probs, -1)
                    _, emotion_pred = torch.max(emotion_probs, -1)
                    
                    # 记录结果
                    batch_strategy_preds.append(strategy_pred.item())
                    batch_emotion_preds.append(emotion_pred.item())
                    batch_strategy_probs.append(strategy_probs.cpu().numpy())
                    batch_emotion_probs.append(emotion_probs.cpu().numpy())
                    
                    # 记录下属模型的输出用于额外评估
                    if sub_model_outputs[i]:
                        self._update_sub_model_evaluations(sub_model_evaluations, sub_model_outputs[i])
                
                # 累加到全局结果
                all_strategy_preds.extend(batch_strategy_preds)
                all_emotion_preds.extend(batch_emotion_preds)
                all_strategy_labels.extend(strategy_labels.cpu().numpy())
                all_emotion_labels.extend(emotion_labels.cpu().numpy())
                all_strategy_probs.extend(batch_strategy_probs)
                all_emotion_probs.extend(batch_emotion_probs)
        
        # 计算评估指标
        metrics = {
            'strategy': self._calculate_metrics(all_strategy_labels, all_strategy_preds, all_strategy_probs, self.strategy_labels),
            'emotion': self._calculate_metrics(all_emotion_labels, all_emotion_preds, all_emotion_probs, self.emotion_labels),
            'sub_model': self._evaluate_sub_model_performance(sub_model_evaluations)
        }
        
        # 计算整体指标
        metrics['overall'] = self._calculate_overall_metrics(metrics)
        
        # 保存评估结果
        if save_results:
            self._save_evaluation_results(metrics, prefix)
            self._generate_visualizations(all_strategy_labels, all_strategy_preds, all_emotion_labels, all_emotion_preds, prefix)
        
        # 打印评估结果
        if self.config['verbose']:
            self._print_evaluation_results(metrics)
        
        return metrics
    
    def _update_sub_model_evaluations(self, sub_model_evaluations, sub_model_output):
        """
        更新下属模型的评估结果
        """
        for model_name, outputs in sub_model_output.items():
            if model_name not in sub_model_evaluations:
                sub_model_evaluations[model_name] = {}
            
            for output_type, output_value in outputs.items():
                if output_type not in sub_model_evaluations[model_name]:
                    sub_model_evaluations[model_name][output_type] = []
                
                # 根据输出类型处理
                if isinstance(output_value, (list, tuple, np.ndarray)):
                    sub_model_evaluations[model_name][output_type].extend(output_value)
                else:
                    sub_model_evaluations[model_name][output_type].append(output_value)
    
    def _evaluate_sub_model_performance(self, sub_model_evaluations):
        """
        评估下属模型的性能
        """
        results = {}
        
        for model_name, outputs in sub_model_evaluations.items():
            results[model_name] = {}
            
            # 检查是否有情感相关的输出
            if 'emotion_pred' in outputs and 'emotion_label' in outputs:
                emotion_preds = outputs['emotion_pred']
                emotion_labels = outputs['emotion_label']
                
                # 计算情感准确率
                emotion_accuracy = accuracy_score(emotion_labels, emotion_preds)
                results[model_name]['emotion_accuracy'] = emotion_accuracy
                
                # 计算情感F1分数
                emotion_f1 = f1_score(emotion_labels, emotion_preds, average='weighted')
                results[model_name]['emotion_f1'] = emotion_f1
            
            # 检查是否有置信度相关的输出
            if 'confidence' in outputs:
                confidences = outputs['confidence']
                results[model_name]['avg_confidence'] = np.mean(confidences)
                results[model_name]['confidence_std'] = np.std(confidences)
            
            # 检查是否有处理时间相关的输出
            if 'processing_time' in outputs:
                processing_times = outputs['processing_time']
                results[model_name]['avg_processing_time'] = np.mean(processing_times)
                results[model_name]['processing_time_std'] = np.std(processing_times)
        
        return results
    
    def _calculate_metrics(self, true_labels, predictions, probabilities, labels):
        """
        计算分类指标
        """
        # 计算准确率
        accuracy = accuracy_score(true_labels, predictions)
        
        # 计算精确率、召回率和F1分数（加权平均）
        precision_weighted = precision_score(true_labels, predictions, average='weighted', zero_division=0)
        recall_weighted = recall_score(true_labels, predictions, average='weighted', zero_division=0)
        f1_weighted = f1_score(true_labels, predictions, average='weighted', zero_division=0)
        
        # 计算精确率、召回率和F1分数（宏平均）
        precision_macro = precision_score(true_labels, predictions, average='macro', zero_division=0)
        recall_macro = recall_score(true_labels, predictions, average='macro', zero_division=0)
        f1_macro = f1_score(true_labels, predictions, average='macro', zero_division=0)
        
        # 计算每个类别的指标
        class_precision, class_recall, class_f1, class_support = precision_recall_fscore_support(
            true_labels, predictions, average=None, zero_division=0
        )
        
        # 创建分类报告
        class_report = classification_report(true_labels, predictions, target_names=labels, output_dict=True, zero_division=0)
        
        # 创建混淆矩阵
        conf_matrix = confusion_matrix(true_labels, predictions)
        
        return {
            'accuracy': accuracy,
            'precision_weighted': precision_weighted,
            'recall_weighted': recall_weighted,
            'f1_weighted': f1_weighted,
            'precision_macro': precision_macro,
            'recall_macro': recall_macro,
            'f1_macro': f1_macro,
            'class_precision': class_precision.tolist(),
            'class_recall': class_recall.tolist(),
            'class_f1': class_f1.tolist(),
            'class_support': class_support.tolist(),
            'confusion_matrix': conf_matrix.tolist(),
            'classification_report': class_report
        }
    
    def _calculate_overall_metrics(self, metrics):
        """
        计算整体评估指标
        """
        # 计算策略和情感的平均指标
        avg_accuracy = (metrics['strategy']['accuracy'] + metrics['emotion']['accuracy']) / 2
        avg_f1_weighted = (metrics['strategy']['f1_weighted'] + metrics['emotion']['f1_weighted']) / 2
        avg_f1_macro = (metrics['strategy']['f1_macro'] + metrics['emotion']['f1_macro']) / 2
        
        # 检查是否有下属模型的评估结果
        sub_model_count = len(metrics['sub_model']) if metrics['sub_model'] else 0
        avg_sub_model_accuracy = 0
        avg_sub_model_confidence = 0
        
        if sub_model_count > 0:
            sub_model_accuracies = []
            sub_model_confidences = []
            
            for model_name, model_metrics in metrics['sub_model'].items():
                if 'emotion_accuracy' in model_metrics:
                    sub_model_accuracies.append(model_metrics['emotion_accuracy'])
                if 'avg_confidence' in model_metrics:
                    sub_model_confidences.append(model_metrics['avg_confidence'])
            
            avg_sub_model_accuracy = np.mean(sub_model_accuracies) if sub_model_accuracies else 0
            avg_sub_model_confidence = np.mean(sub_model_confidences) if sub_model_confidences else 0
        
        return {
            'avg_accuracy': avg_accuracy,
            'avg_f1_weighted': avg_f1_weighted,
            'avg_f1_macro': avg_f1_macro,
            'sub_model_count': sub_model_count,
            'avg_sub_model_accuracy': avg_sub_model_accuracy,
            'avg_sub_model_confidence': avg_sub_model_confidence
        }
    
    def _save_evaluation_results(self, metrics, prefix='eval'):
        """
        保存评估结果
        """
        # 生成时间戳
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 保存完整指标
        metrics_path = os.path.join(self.config['metrics_dir'], f'{prefix}_metrics_{timestamp}.json')
        with open(metrics_path, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Evaluation metrics saved to {metrics_path}")
        
        # 保存摘要指标
        summary = {
            'timestamp': timestamp,
            'overall': metrics['overall'],
            'strategy_accuracy': metrics['strategy']['accuracy'],
            'emotion_accuracy': metrics['emotion']['accuracy'],
            'strategy_f1': metrics['strategy']['f1_weighted'],
            'emotion_f1': metrics['emotion']['f1_weighted']
        }
        
        summary_path = os.path.join(self.config['output_dir'], f'{prefix}_summary_{timestamp}.json')
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Evaluation summary saved to {summary_path}")
    
    def _generate_visualizations(self, strategy_labels, strategy_preds, emotion_labels, emotion_preds, prefix='eval'):
        """
        生成可视化图表
        """
        if not self.config['save_plots']:
            return
        
        # 生成时间戳
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 生成策略混淆矩阵
        self._plot_confusion_matrix(
            strategy_labels, strategy_preds, self.strategy_labels,
            title='Strategy Confusion Matrix',
            filename=os.path.join(self.config['figures_dir'], f'{prefix}_strategy_confusion_matrix_{timestamp}.png')
        )
        
        # 生成情感混淆矩阵
        self._plot_confusion_matrix(
            emotion_labels, emotion_preds, self.emotion_labels,
            title='Emotion Confusion Matrix',
            filename=os.path.join(self.config['figures_dir'], f'{prefix}_emotion_confusion_matrix_{timestamp}.png')
        )
        
        # 生成指标对比图
        self._plot_metrics_comparison(
            [
                {'name': 'Strategy Accuracy', 'value': accuracy_score(strategy_labels, strategy_preds)},
                {'name': 'Emotion Accuracy', 'value': accuracy_score(emotion_labels, emotion_preds)},
                {'name': 'Strategy F1', 'value': f1_score(strategy_labels, strategy_preds, average='weighted')},
                {'name': 'Emotion F1', 'value': f1_score(emotion_labels, emotion_preds, average='weighted')}
            ],
            title='Performance Metrics Comparison',
            filename=os.path.join(self.config['figures_dir'], f'{prefix}_metrics_comparison_{timestamp}.png')
        )
    
    def _plot_confusion_matrix(self, true_labels, predictions, labels, title='Confusion Matrix', filename=None):
        """
        绘制混淆矩阵
        """
        try:
            cm = confusion_matrix(true_labels, predictions)
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
            plt.title(title)
            plt.xlabel('Predicted')
            plt.ylabel('True')
            
            if filename:
                plt.savefig(filename, dpi=300, bbox_inches='tight')
                logger.info(f"Confusion matrix saved to {filename}")
            
            plt.close()
        except Exception as e:
            logger.error(f"Failed to generate confusion matrix: {str(e)}")
    
    def _plot_metrics_comparison(self, metrics, title='Metrics Comparison', filename=None):
        """
        绘制指标对比图
        """
        try:
            plt.figure(figsize=(10, 6))
            
            names = [m['name'] for m in metrics]
            values = [m['value'] for m in metrics]
            
            bars = plt.bar(names, values, color='skyblue')
            plt.title(title)
            plt.ylabel('Score')
            plt.ylim(0, 1.1)
            
            # 添加数值标签
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                         f'{height:.4f}', ha='center', va='bottom')
            
            # 旋转x轴标签以避免重叠
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            
            if filename:
                plt.savefig(filename, dpi=300, bbox_inches='tight')
                logger.info(f"Metrics comparison plot saved to {filename}")
            
            plt.close()
        except Exception as e:
            logger.error(f"Failed to generate metrics comparison plot: {str(e)}")
    
    def _print_evaluation_results(self, metrics):
        """
        打印评估结果
        """
        print("\n=== Evaluation Results ===")
        
        # 打印整体指标
        print("\nOverall Metrics:")
        print(f"Average Accuracy: {metrics['overall']['avg_accuracy']:.4f}")
        print(f"Average F1 Score (Weighted): {metrics['overall']['avg_f1_weighted']:.4f}")
        print(f"Average F1 Score (Macro): {metrics['overall']['avg_f1_macro']:.4f}")
        
        # 打印策略指标
        print("\nStrategy Metrics:")
        print(f"Accuracy: {metrics['strategy']['accuracy']:.4f}")
        print(f"F1 Score (Weighted): {metrics['strategy']['f1_weighted']:.4f}")
        print(f"F1 Score (Macro): {metrics['strategy']['f1_macro']:.4f}")
        
        # 打印情感指标
        print("\nEmotion Metrics:")
        print(f"Accuracy: {metrics['emotion']['accuracy']:.4f}")
        print(f"F1 Score (Weighted): {metrics['emotion']['f1_weighted']:.4f}")
        print(f"F1 Score (Macro): {metrics['emotion']['f1_macro']:.4f}")
        
        # 打印下属模型指标
        if metrics['sub_model']:
            print("\nSub-Model Metrics:")
            for model_name, model_metrics in metrics['sub_model'].items():
                print(f"  {model_name}:")
                if 'emotion_accuracy' in model_metrics:
                    print(f"    Emotion Accuracy: {model_metrics['emotion_accuracy']:.4f}")
                if 'emotion_f1' in model_metrics:
                    print(f"    Emotion F1: {model_metrics['emotion_f1']:.4f}")
                if 'avg_confidence' in model_metrics:
                    print(f"    Average Confidence: {model_metrics['avg_confidence']:.4f}")
                if 'avg_processing_time' in model_metrics:
                    print(f"    Average Processing Time: {model_metrics['avg_processing_time']:.4f}s")
        
        print("\n=========================")
    
    def evaluate_integration_effectiveness(self, base_performance, enhanced_performance):
        """
        评估情感集成的有效性
        
        参数:
            base_performance: 基础模型性能（没有情感集成）
            enhanced_performance: 增强模型性能（有情感集成）
        
        返回:
            集成有效性评估结果
        """
        # 计算性能提升
        improvement = {
            'accuracy_improvement': enhanced_performance['overall']['avg_accuracy'] - base_performance['overall']['avg_accuracy'],
            'f1_improvement': enhanced_performance['overall']['avg_f1_weighted'] - base_performance['overall']['avg_f1_weighted'],
            'strategy_accuracy_improvement': enhanced_performance['strategy']['accuracy'] - base_performance['strategy']['accuracy'],
            'emotion_accuracy_improvement': enhanced_performance['emotion']['accuracy'] - base_performance['emotion']['accuracy'],
        }
        
        # 判断集成是否有效
        is_effective = improvement['accuracy_improvement'] > 0 or improvement['f1_improvement'] > 0
        
        # 保存集成有效性评估结果
        results = {
            'base_performance': base_performance['overall'],
            'enhanced_performance': enhanced_performance['overall'],
            'improvement': improvement,
            'is_effective': is_effective
        }
        
        # 生成时间戳
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 保存结果
        integration_path = os.path.join(self.config['output_dir'], f'integration_effectiveness_{timestamp}.json')
        with open(integration_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Integration effectiveness results saved to {integration_path}")
        
        # 打印集成有效性评估结果
        if self.config['verbose']:
            print("\n=== Integration Effectiveness Evaluation ===")
            print(f"Base Accuracy: {base_performance['overall']['avg_accuracy']:.4f}")
            print(f"Enhanced Accuracy: {enhanced_performance['overall']['avg_accuracy']:.4f}")
            print(f"Accuracy Improvement: {improvement['accuracy_improvement']:.4f} ({improvement['accuracy_improvement']*100:.2f}%)")
            print(f"\nBase F1 Score: {base_performance['overall']['avg_f1_weighted']:.4f}")
            print(f"Enhanced F1 Score: {enhanced_performance['overall']['avg_f1_weighted']:.4f}")
            print(f"F1 Score Improvement: {improvement['f1_improvement']:.4f} ({improvement['f1_improvement']*100:.2f}%)")
            print(f"\nIntegration {'is effective' if is_effective else 'is not effective'}")
            print("\n=========================================")
        
        return results
    
    def get_error_analysis(self, data_loader, max_errors=100):
        """
        进行错误分析，找出模型容易出错的地方
        
        参数:
            data_loader: 数据加载器
            max_errors: 最大错误样本数
        
        返回:
            错误分析结果
        """
        strategy_errors = []
        emotion_errors = []
        combined_errors = []
        
        with torch.no_grad():
            for batch in data_loader:
                features = batch['features']
                strategy_labels = batch['strategy_labels']
                emotion_labels = batch['emotion_labels']
                sub_model_outputs = batch['sub_model_outputs']
                
                for i in range(len(features)):
                    # 前向传播
                    strategy_probs, emotion_probs = self.model(features[i], sub_model_outputs[i])
                    
                    # 获取预测结果
                    _, strategy_pred = torch.max(strategy_probs, -1)
                    _, emotion_pred = torch.max(emotion_probs, -1)
                    
                    # 记录错误
                    strategy_error = strategy_pred.item() != strategy_labels[i].item()
                    emotion_error = emotion_pred.item() != emotion_labels[i].item()
                    
                    if strategy_error:
                        error_sample = {
                            'feature': features[i],
                            'true_strategy': strategy_labels[i].item(),
                            'pred_strategy': strategy_pred.item(),
                            'strategy_prob': strategy_probs.tolist(),
                            'true_emotion': emotion_labels[i].item(),
                            'pred_emotion': emotion_pred.item(),
                            'sub_model_output': sub_model_outputs[i]
                        }
                        
                        strategy_errors.append(error_sample)
                    
                    if emotion_error:
                        error_sample = {
                            'feature': features[i],
                            'true_strategy': strategy_labels[i].item(),
                            'pred_strategy': strategy_pred.item(),
                            'true_emotion': emotion_labels[i].item(),
                            'pred_emotion': emotion_pred.item(),
                            'emotion_prob': emotion_probs.tolist(),
                            'sub_model_output': sub_model_outputs[i]
                        }
                        
                        emotion_errors.append(error_sample)
                    
                    if strategy_error and emotion_error:
                        combined_errors.append(error_sample)
                    
                    # 如果已经收集了足够的错误样本，提前结束
                    if len(strategy_errors) >= max_errors and len(emotion_errors) >= max_errors:
                        break
                
                # 如果已经收集了足够的错误样本，提前结束
                if len(strategy_errors) >= max_errors and len(emotion_errors) >= max_errors:
                    break
        
        # 分析错误模式
        analysis = {
            'strategy_error_count': len(strategy_errors),
            'emotion_error_count': len(emotion_errors),
            'combined_error_count': len(combined_errors),
            'strategy_error_examples': strategy_errors[:10],  # 只保存前10个例子
            'emotion_error_examples': emotion_errors[:10],  # 只保存前10个例子
            'error_distribution': self._analyze_error_distribution(strategy_errors, emotion_errors)
        }
        
        # 生成时间戳
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 保存错误分析结果
        error_analysis_path = os.path.join(self.config['output_dir'], f'error_analysis_{timestamp}.json')
        with open(error_analysis_path, 'w', encoding='utf-8') as f:
            json.dump(analysis, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Error analysis saved to {error_analysis_path}")
        
        return analysis
    
    def _analyze_error_distribution(self, strategy_errors, emotion_errors):
        """
        分析错误分布
        """
        # 统计策略错误分布
        strategy_error_dist = {}
        for error in strategy_errors:
            true_label = error['true_strategy']
            pred_label = error['pred_strategy']
            key = f"{true_label}->{pred_label}"
            strategy_error_dist[key] = strategy_error_dist.get(key, 0) + 1
        
        # 统计情感错误分布
        emotion_error_dist = {}
        for error in emotion_errors:
            true_label = error['true_emotion']
            pred_label = error['pred_emotion']
            key = f"{true_label}->{pred_label}"
            emotion_error_dist[key] = emotion_error_dist.get(key, 0) + 1
        
        return {
            'strategy_error_distribution': strategy_error_dist,
            'emotion_error_distribution': emotion_error_dist
        }

# 工具函数
def create_evaluator(model, config_path=None):
    """
    创建评估器实例
    
    参数:
        model: 要评估的模型
        config_path: 配置文件路径
    
    返回:
        ModelEvaluator实例
    """
    config = {}
    
    # 如果提供了配置文件路径，加载配置
    if config_path and os.path.exists(config_path):
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            logger.info(f"Evaluator config loaded from {config_path}")
        except Exception as e:
            logger.error(f"Failed to load evaluator config from {config_path}: {str(e)}")
    
    # 创建评估器实例
    evaluator = ModelEvaluator(model, config)
    
    return evaluator

def compare_models(model1, model2, data_loader, config=None):
    """
    比较两个模型的性能
    
    参数:
        model1: 第一个模型
        model2: 第二个模型
        data_loader: 数据加载器
        config: 评估配置
    
    返回:
        模型比较结果
    """
    # 创建评估器
    evaluator1 = ModelEvaluator(model1, config)
    evaluator2 = ModelEvaluator(model2, config)
    
    # 评估两个模型
    metrics1 = evaluator1.evaluate(data_loader, save_results=False, prefix='model1')
    metrics2 = evaluator2.evaluate(data_loader, save_results=False, prefix='model2')
    
    # 比较结果
    comparison = {
        'model1': metrics1['overall'],
        'model2': metrics2['overall'],
        'difference': {
            'accuracy': metrics2['overall']['avg_accuracy'] - metrics1['overall']['avg_accuracy'],
            'f1_weighted': metrics2['overall']['avg_f1_weighted'] - metrics1['overall']['avg_f1_weighted'],
            'f1_macro': metrics2['overall']['avg_f1_macro'] - metrics1['overall']['avg_f1_macro']
        },
        'better_model': 'model2' if metrics2['overall']['avg_accuracy'] > metrics1['overall']['avg_accuracy'] else 'model1'
    }
    
    # 生成时间戳
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # 保存比较结果
    comparison_path = os.path.join(config.get('output_dir', './evaluation_results'), f'model_comparison_{timestamp}.json')
    with open(comparison_path, 'w', encoding='utf-8') as f:
        json.dump(comparison, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Model comparison results saved to {comparison_path}")
    
    # 打印比较结果
    print("\n=== Model Comparison Results ===")
    print(f"Model 1 - Accuracy: {metrics1['overall']['avg_accuracy']:.4f}, F1: {metrics1['overall']['avg_f1_weighted']:.4f}")
    print(f"Model 2 - Accuracy: {metrics2['overall']['avg_accuracy']:.4f}, F1: {metrics2['overall']['avg_f1_weighted']:.4f}")
    print(f"Difference - Accuracy: {comparison['difference']['accuracy']:.4f}, F1: {comparison['difference']['f1_weighted']:.4f}")
    print(f"Better Model: {comparison['better_model']}")
    print("\n===============================")
    
    return comparison

def analyze_model_behavior(model, data_loader, config=None):
    """
    分析模型行为特征
    
    参数:
        model: 要分析的模型
        data_loader: 数据加载器
        config: 分析配置
    
    返回:
        模型行为分析结果
    """
    # 创建评估器
    evaluator = ModelEvaluator(model, config)
    
    # 评估模型
    metrics = evaluator.evaluate(data_loader, save_results=False)
    
    # 进行错误分析
    error_analysis = evaluator.get_error_analysis(data_loader)
    
    # 综合分析结果
    analysis = {
        'performance_metrics': metrics,
        'error_analysis': error_analysis,
        'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S')
    }
    
    # 保存分析结果
    analysis_path = os.path.join(config.get('output_dir', './evaluation_results'), f'model_behavior_analysis_{analysis["timestamp"]}.json')
    with open(analysis_path, 'w', encoding='utf-8') as f:
        json.dump(analysis, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Model behavior analysis saved to {analysis_path}")
    
    return analysis

# 主函数演示
if __name__ == "__main__":
    from enhanced_manager import ManagementModel
    from enhanced_trainer import ManagementDataset, DataLoader
    
    # 创建模型
    model = ManagementModel()
    
    # 创建评估器
    evaluator = ModelEvaluator(model)
    
    # 注意：这里只是演示，实际使用时需要提供真实的数据
    print("This is a demo of the enhanced evaluator for A_management model.")
    print("In a real scenario, you would create a dataset and data loader with valid data.")
    print("Then call evaluator.evaluate(data_loader) to evaluate the model.")