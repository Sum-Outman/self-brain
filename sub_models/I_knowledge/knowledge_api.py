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

# 知识库专家模型API服务
# Knowledge Base Expert Model API Service

from flask import Flask, jsonify, request
import threading
import time
import random
from datetime import datetime
import json
import os
import re
import hashlib
from collections import defaultdict

app = Flask(__name__)

# 支持的学科领域
SUPPORTED_DISCIPLINES = [
    "physics", "mathematics", "chemistry", "medicine", "law", 
    "history", "sociology", "humanities", "psychology", "economics",
    "management", "mechanical_engineering", "electronic_engineering",
    "food_engineering", "chemical_engineering"
]

# 知识库模型状态
knowledge_status = {
    "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "knowledge_entries": 0,
    "queries_processed": 0,
    "learning_operations": 0,
    "tutoring_sessions": 0,
    "verifications_performed": 0,
    "recommendations_made": 0,
    "active_assistance_requests": 0,  # 新增主动辅助请求计数器
    "current_version": "1.0.0",
    "disciplines_supported": SUPPORTED_DISCIPLINES
}

# 知识库存储
KNOWLEDGE_BASE_FILE = "knowledge_base.json"
knowledge_base = defaultdict(lambda: defaultdict(list))  # discipline -> topic -> [knowledge entries]

def load_knowledge_base():
    """加载知识库"""
    global knowledge_base
    if os.path.exists(KNOWLEDGE_BASE_FILE):
        try:
            with open(KNOWLEDGE_BASE_FILE, 'r', encoding='utf-8') as f:
                knowledge_base = json.load(f)
            print("知识库已加载")
            # 更新知识条目计数
            count = 0
            for discipline in knowledge_base:
                for topic in knowledge_base[discipline]:
                    count += len(knowledge_base[discipline][topic])
            knowledge_status["knowledge_entries"] = count
        except Exception as e:
            print(f"加载知识库失败: {str(e)}")
    else:
        print("使用空知识库")

def save_knowledge_base():
    """保存知识库到文件"""
    try:
        with open(KNOWLEDGE_BASE_FILE, 'w', encoding='utf-8') as f:
            json.dump(knowledge_base, f, ensure_ascii=False, indent=2)
        print("知识库已保存")
    except Exception as e:
        print(f"保存知识库失败: {str(e)}")

def knowledge_maintenance():
    """定期维护知识库模型"""
    while True:
        # 每30分钟自动保存一次
        save_knowledge_base()
        time.sleep(1800)

def generate_knowledge_id(discipline, topic, content):
    """生成知识条目ID"""
    return hashlib.md5(f"{discipline}_{topic}_{content[:50]}".encode('utf-8')).hexdigest()

@app.route('/health', methods=['GET'])
def health_check():
    """健康检查端点"""
    return jsonify({
        "status": "healthy", 
        "timestamp": time.time(),
        "knowledge_entries": knowledge_status["knowledge_entries"],
        "version": knowledge_status["current_version"]
    })

@app.route('/get_status', methods=['GET'])
def get_status():
    """获取知识库模型状态"""
    return jsonify({
        "status": "success",
        "status_info": {
            "last_updated": knowledge_status["last_updated"],
            "knowledge_entries": knowledge_status["knowledge_entries"],
            "queries_processed": knowledge_status["queries_processed"],
            "learning_operations": knowledge_status["learning_operations"],
            "tutoring_sessions": knowledge_status["tutoring_sessions"],
            "verifications_performed": knowledge_status["verifications_performed"],
        "recommendations_made": knowledge_status["recommendations_made"],
        "active_assistance_requests": knowledge_status.get("active_assistance_requests", 0),
        "current_version": knowledge_status["current_version"],
        "disciplines_supported": knowledge_status["disciplines_supported"]
        }
    })

@app.route('/query_knowledge', methods=['POST'])
def query_knowledge():
    """查询知识库"""
    data = request.json
    discipline = data.get('discipline', "")
    topic = data.get('topic', "")
    keywords = data.get('keywords', [])
    max_results = data.get('max_results', 10)
    
    knowledge_status["queries_processed"] += 1
    
    results = []
    
    # 如果没有指定学科，搜索所有学科
    disciplines = [discipline] if discipline else SUPPORTED_DISCIPLINES
    
    for disc in disciplines:
        # 如果没有指定主题，搜索该学科的所有主题
        topics = [topic] if topic else list(knowledge_base.get(disc, {}).keys())
        
        for top in topics:
            if disc in knowledge_base and top in knowledge_base[disc]:
                for entry in knowledge_base[disc][top]:
                    # 检查关键词匹配
                    match = True
                    if keywords:
                        content = entry["content"].lower()
                        for kw in keywords:
                            if kw.lower() not in content:
                                match = False
                                break
                    
                    if match:
                        results.append({
                            "discipline": disc,
                            "topic": top,
                            **entry
                        })
                        if len(results) >= max_results:
                            break
                if len(results) >= max_results:
                    break
        if len(results) >= max_results:
            break
    
    return jsonify({
        "status": "success",
        "results": results,
        "count": len(results)
    })

@app.route('/learn_knowledge', methods=['POST'])
def learn_knowledge():
    """学习新知识"""
    data = request.json
    if not data or 'discipline' not in data or 'topic' not in data or 'content' not in data:
        return jsonify({"status": "error", "message": "缺少学科、主题或内容参数"}), 400
    
    discipline = data['discipline']
    topic = data['topic']
    content = data['content']
    source = data.get('source', "unknown")
    reliability = data.get('reliability', 0.8)  # 默认可靠性0.8
    
    if discipline not in SUPPORTED_DISCIPLINES:
        return jsonify({"status": "error", "message": f"不支持的学科: {discipline}"}), 400
    
    # 创建知识条目
    entry_id = generate_knowledge_id(discipline, topic, content)
    knowledge_entry = {
        "id": entry_id,
        "content": content,
        "source": source,
        "reliability": reliability,
        "created_at": datetime.now().isoformat(),
        "last_verified": datetime.now().isoformat()
    }
    
    # 存储到知识库
    if discipline not in knowledge_base:
        knowledge_base[discipline] = {}
    if topic not in knowledge_base[discipline]:
        knowledge_base[discipline][topic] = []
    
    knowledge_base[discipline][topic].append(knowledge_entry)
    
    # 更新状态
    knowledge_status["knowledge_entries"] += 1
    knowledge_status["learning_operations"] += 1
    
    return jsonify({
        "status": "success",
        "message": "知识条目已添加",
        "knowledge_entry": knowledge_entry
    })

@app.route('/tutor', methods=['POST'])
def tutor():
    """提供教学辅导"""
    data = request.json
    if not data or 'question' not in data:
        return jsonify({"status": "error", "message": "缺少问题参数"}), 400
    
    question = data['question']
    context = data.get('context', "")
    max_depth = data.get('max_depth', 2)  # 解释深度
    
    knowledge_status["tutoring_sessions"] += 1
    
    # 模拟教学辅导
    explanation = f"关于'{question}'的解释：\n"
    explanation += "1. 这是一个重要的概念，涉及多个方面。\n"
    explanation += "2. 首先，我们需要理解基本定义。\n"
    explanation += "3. 其次，考虑其在实际应用中的表现。\n"
    explanation += "4. 最后，分析相关案例和研究。"
    
    related_topics = ["相关主题1", "相关主题2", "相关主题3"]
    
    return jsonify({
        "status": "success",
        "question": question,
        "explanation": explanation,
        "related_topics": related_topics
    })

@app.route('/verify_knowledge', methods=['POST'])
def verify_knowledge():
    """验证知识准确性"""
    data = request.json
    if not data or 'knowledge_id' not in data:
        return jsonify({"status": "error", "message": "缺少知识ID参数"}), 400
    
    knowledge_id = data['knowledge_id']
    verification_method = data.get('verification_method', "cross_reference")
    
    knowledge_status["verifications_performed"] += 1
    
    # 模拟知识验证
    verification_result = {
        "status": "verified",
        "confidence": 0.95,
        "verification_method": verification_method,
        "details": "通过多个可靠来源交叉验证",
        "last_verified": datetime.now().isoformat()
    }
    
    # 更新知识条目中的验证状态
    for disc in knowledge_base:
        for topic in knowledge_base[disc]:
            for entry in knowledge_base[disc][topic]:
                if entry["id"] == knowledge_id:
                    entry["last_verified"] = verification_result["last_verified"]
                    entry["reliability"] = min(1.0, entry.get("reliability", 0.8) + 0.05)
    
    return jsonify({
        "status": "success",
        "verification_result": verification_result
    })

@app.route('/recommend', methods=['POST'])
def recommend():
    """推荐相关知识"""
    data = request.json
    context = data.get('context', "")
    current_topic = data.get('current_topic', "")
    max_recommendations = data.get('max_recommendations', 5)
    
    knowledge_status["recommendations_made"] += 1
    
    # 模拟知识推荐
    recommendations = []
    for i in range(min(3, max_recommendations)):
        recommendations.append({
            "topic": f"相关主题{i+1}",
            "discipline": random.choice(SUPPORTED_DISCIPLINES),
            "reason": f"与'{current_topic}'高度相关",
            "confidence": random.uniform(0.7, 0.95)
        })
    
    return jsonify({
        "status": "success",
        "recommendations": recommendations
    })

if __name__ == '__main__':
    # 加载知识库
    load_knowledge_base()
    
    # 启动知识库模型维护线程
    maintenance_thread = threading.Thread(target=knowledge_maintenance, daemon=True)
    maintenance_thread.start()
    
    # 启动API服务
    app.run(host='0.0.0.0', port=5008)
