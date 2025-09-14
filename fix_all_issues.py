#!/usr/bin/env python3
"""
修复所有知识管理系统功能缺陷的脚本
"""
import os
import json
import re

def fix_api_issues():
    """修复API端点问题"""
    app_file = "d:/shiyan/web_interface/app.py"
    
    # 读取文件内容
    with open(app_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 修复1: 删除重复的批量删除API
    # 查找并删除第2个重复的定义
    delete_selected_pattern = r'@app\.route\(\'/api/knowledge/delete_selected\', methods=\[\'POST\'\]\)\s*\n\s*def delete_selected_knowledge\([^)]*\):[^@]*?(?=@app\.route|$)'
    
    # 查找所有匹配
    matches = list(re.finditer(delete_selected_pattern, content, re.DOTALL))
    if len(matches) > 1:
        # 保留第一个，删除其余的
        first_end = matches[0].end()
        second_start = matches[1].start()
        content = content[:first_end] + content[second_start:]
    
    # 修复2: 修复知识获取API的数据格式
    get_knowledge_pattern = r'def get_knowledge_item\([^)]*\):.*?return jsonify\([^)]*\)'
    new_get_knowledge = '''def get_knowledge_item(knowledge_id):
    """获取单个知识条目详情"""
    try:
        knowledge_path = os.path.join('knowledge_base_storage', knowledge_id)
        
        if not os.path.exists(knowledge_path):
            return jsonify({'success': False, 'message': 'Knowledge not found'}), 404
        
        # 读取内容
        content_file = os.path.join(knowledge_path, 'content.txt')
        info_file = os.path.join(knowledge_path, 'info.json')
        
        # 读取元数据
        knowledge_data = {
            'id': knowledge_id,
            'title': '',
            'content': '',
            'category': '',
            'summary': '',
            'tags': [],
            'created_at': '',
            'updated_at': '',
            'file_path': knowledge_path
        }
        
        # 读取内容
        if os.path.exists(content_file):
            with open(content_file, 'r', encoding='utf-8') as f:
                knowledge_data['content'] = f.read()
        
        # 读取元数据
        if os.path.exists(info_file):
            with open(info_file, 'r', encoding='utf-8') as f:
                info = json.load(f)
                knowledge_data.update({
                    'title': info.get('title', ''),
                    'category': info.get('category', ''),
                    'summary': info.get('summary', ''),
                    'tags': info.get('tags', []),
                    'created_at': info.get('created_at', ''),
                    'updated_at': info.get('updated_at', '')
                })
        
        return jsonify({
            'success': True,
            'data': knowledge_data
        })
        
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500'''
    
    # 修复3: 修复导出API
    export_pattern = r'@app\.route\(\'/api/export_knowledge\'\)\s*\n\s*def export_knowledge\([^)]*\):[^@]*?(?=@app\.route|$)'
    new_export = '''@app.route('/api/export_knowledge')
def export_knowledge():
    """导出所有知识条目"""
    try:
        storage_path = 'knowledge_base_storage'
        knowledge_list = []
        
        if os.path.exists(storage_path):
            for knowledge_id in os.listdir(storage_path):
                knowledge_path = os.path.join(storage_path, knowledge_id)
                if os.path.isdir(knowledge_path):
                    knowledge_data = {'id': knowledge_id}
                    
                    # 读取内容
                    content_file = os.path.join(knowledge_path, 'content.txt')
                    info_file = os.path.join(knowledge_path, 'info.json')
                    
                    if os.path.exists(content_file):
                        with open(content_file, 'r', encoding='utf-8') as f:
                            knowledge_data['content'] = f.read()
                    
                    if os.path.exists(info_file):
                        with open(info_file, 'r', encoding='utf-8') as f:
                            info = json.load(f)
                            knowledge_data.update(info)
                    
                    knowledge_list.append(knowledge_data)
        
        return jsonify({
            'success': True,
            'knowledge_base': knowledge_list,
            'total_count': len(knowledge_list)
        })
        
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500'''
    
    # 应用修复
    content = re.sub(get_knowledge_pattern, new_get_knowledge, content, flags=re.DOTALL)
    content = re.sub(export_pattern, new_export, content, flags=re.DOTALL)
    
    # 写入修复后的内容
    with open(app_file, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("✅ API端点问题已修复")

def fix_search_api():
    """修复搜索API"""
    app_file = "d:/shiyan/web_interface/app.py"
    
    with open(app_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 添加搜索API
    search_api = '''
@app.route('/api/knowledge/search')
def search_knowledge():
    """搜索知识条目"""
    try:
        query = request.args.get('q', '')
        category = request.args.get('category', '')
        
        storage_path = 'knowledge_base_storage'
        results = []
        
        if not query:
            return jsonify({'success': True, 'results': [], 'total': 0})
        
        if os.path.exists(storage_path):
            for knowledge_id in os.listdir(storage_path):
                knowledge_path = os.path.join(storage_path, knowledge_id)
                if os.path.isdir(knowledge_path):
                    # 读取内容
                    content_file = os.path.join(knowledge_path, 'content.txt')
                    info_file = os.path.join(knowledge_path, 'info.json')
                    
                    knowledge_data = {'id': knowledge_id}
                    
                    if os.path.exists(content_file):
                        with open(content_file, 'r', encoding='utf-8') as f:
                            knowledge_data['content'] = f.read()
                    
                    if os.path.exists(info_file):
                        with open(info_file, 'r', encoding='utf-8') as f:
                            info = json.load(f)
                            knowledge_data.update(info)
                    
                    # 搜索匹配
                    search_text = f"{knowledge_data.get('title', '')} {knowledge_data.get('content', '')} {knowledge_data.get('category', '')}".lower()
                    
                    if query.lower() in search_text:
                        if not category or category == knowledge_data.get('category', ''):
                            results.append(knowledge_data)
        
        return jsonify({
            'success': True,
            'results': results,
            'total': len(results)
        })
        
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500
'''
    
    # 确保搜索API已添加
    if '@app.route(\'/api/knowledge/search\')' not in content:
        # 找到合适的位置插入
        lines = content.split('\n')
        insert_index = -1
        for i, line in enumerate(lines):
            if '@app.route(\'/api/export_knowledge\')' in line:
                insert_index = i
                break
        
        if insert_index > 0:
            lines.insert(insert_index, search_api)
            content = '\n'.join(lines)
            with open(app_file, 'w', encoding='utf-8') as f:
                f.write(content)
            print("✅ 搜索API已添加")

def main():
    """执行所有修复"""
    print("开始修复知识管理系统功能缺陷...")
    
    fix_api_issues()
    fix_search_api()
    
    print("🎉 所有修复完成！")

if __name__ == "__main__":
    main()