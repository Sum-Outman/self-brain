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

# 知识模型API服务
# Knowledge Model API Service

from flask import Flask, request, jsonify
from .model import KnowledgeModel

app = Flask(__name__)
model = KnowledgeModel()

@app.route('/query', methods=['POST'])
def query_knowledge():
    """查询知识库API | Query knowledge base API"""
    data = request.json
    query = data.get('query')
    
    if not query:
        return jsonify({'error': 'Query is required'}), 400
    
    try:
        result = model.query_knowledge(query)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/add', methods=['POST'])
def add_knowledge():
    """添加知识API | Add knowledge API"""
    data = request.json
    knowledge = data.get('knowledge')
    
    if not knowledge:
        return jsonify({'error': 'Knowledge data is required'}), 400
    
    try:
        result = model.add_knowledge(knowledge)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/update', methods=['POST'])
def update_knowledge():
    """更新知识API | Update knowledge API"""
    data = request.json
    knowledge_id = data.get('id')
    updates = data.get('updates')
    
    if not knowledge_id or not updates:
        return jsonify({'error': 'Knowledge ID and updates are required'}), 400
    
    try:
        result = model.update_knowledge(knowledge_id, updates)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/delete', methods=['POST'])
def delete_knowledge():
    """删除知识API | Delete knowledge API"""
    data = request.json
    knowledge_id = data.get('id')
    
    if not knowledge_id:
        return jsonify({'error': 'Knowledge ID is required'}), 400
    
    try:
        result = model.delete_knowledge(knowledge_id)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/search', methods=['POST'])
def search_knowledge():
    """搜索知识API | Search knowledge API"""
    data = request.json
    keyword = data.get('keyword')
    
    if not keyword:
        return jsonify({'error': 'Keyword is required'}), 400
    
    try:
        result = model.search_knowledge(keyword)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """健康检查 | Health check"""
    return jsonify({"status": "healthy", "model": "I_knowledge"})

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5009))
    app.run(host='0.0.0.0', port=port, debug=True)