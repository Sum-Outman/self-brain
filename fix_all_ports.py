#!/usr/bin/env python3
"""
Self Brain AGI System - 端口统一修正脚本
将所有服务的端口修正为标准配置
"""

import os
import re
import glob

# 标准端口映射
STANDARD_PORTS = {
    5000: ["web_interface/working_enhanced_chat.py", "web_interface/app_fixed.py"],
    5001: ["a_management_server.py", "web_interface/backend/a_manager_api.py", "web_interface/backend/advanced_chat_api.py"],
    5002: ["sub_models/B_language/app.py", "sub_models/B_language/unified_api.py", "web_interface/debug_routes.py"],
    5003: ["sub_models/C_audio/api.py", "sub_models/C_audio/app.py"],
    5004: ["sub_models/D_image/api.py", "sub_models/D_image/app.py"],
    5005: ["sub_models/E_video/api.py", "sub_models/E_video/app.py", "improved_api_server.py"],
    5006: ["sub_models/F_spatial/api.py", "sub_models/F_spatial/app.py", "web_interface/enhanced_ai_chat.py"],
    5007: ["sub_models/G_sensor/api.py", "sub_models/G_sensor/app.py", "sub_models/G_sensor/sensor_api.py"],
    5008: ["sub_models/H_computer_control/api.py", "sub_models/H_computer_control/app.py"],
    5009: ["sub_models/I_knowledge/api.py", "sub_models/I_knowledge/app.py", "sub_models/I_knowledge/knowledge_api.py", 
           "sub_models/I_knowledge/knowledge_app.py", "sub_models/I_knowledge/test_server.py", "sub_models/I_knowledge/minimal_test.py"],
    5010: ["sub_models/J_motion/api.py", "sub_models/J_motion/app.py"],
    5011: ["sub_models/K_programming/programming_api.py"],
    5012: ["training_manager.py"],
    5013: ["quantum_integration.py"],
    5014: ["a_manager_standalone.py"],
    5015: ["manager_model/app.py"],
}

def fix_port_in_file(file_path, target_port):
    """修正文件中的端口配置"""
    if not os.path.exists(file_path):
        return False
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 匹配app.run的端口配置
    patterns = [
        r'app\.run\([^)]*port\s*=\s*\d+[^)]*\)',
        r'app\.run\([^)]*port\s*=\s*\d+',
    ]
    
    original_content = content
    
    for pattern in patterns:
        # 查找并替换端口
        matches = re.findall(pattern, content)
        for match in matches:
            # 提取数字并替换
            port_match = re.search(r'port\s*=\s*(\d+)', match)
            if port_match:
                old_port = port_match.group(1)
                new_match = match.replace(f"port={old_port}", f"port={target_port}")
                content = content.replace(match, new_match)
    
    # 处理其他端口配置格式
    content = re.sub(r'port\s*=\s*\d+', f'port={target_port}', content)
    
    if content != original_content:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"✅ 已修正 {file_path} -> 端口 {target_port}")
        return True
    
    return False

def main():
    """主函数"""
    print("🚀 Self Brain AGI System - 端口统一修正")
    print("=" * 50)
    
    fixed_count = 0
    
    for port, file_list in STANDARD_PORTS.items():
        for file_path in file_list:
            if os.path.exists(file_path):
                if fix_port_in_file(file_path, port):
                    fixed_count += 1
            else:
                print(f"⚠️  文件不存在: {file_path}")
    
    print(f"\n📊 修正完成！共修正 {fixed_count} 个文件")
    print("\n标准端口配置:")
    for port, services in STANDARD_PORTS.items():
        print(f"  端口 {port}: {', '.join([os.path.basename(s) for s in services])}")

if __name__ == "__main__":
    main()