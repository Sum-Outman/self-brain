#!/usr/bin/env python3
"""
图片转SVG脚本
将指定目录中的图片文件转换为SVG格式，包含嵌入的base64图片数据。
"""

import os
import base64
from PIL import Image
import sys

def check_and_install_pillow():
    """检查并安装Pillow库"""
    try:
        from PIL import Image
        print("✓ Pillow已安装")
        return True
    except ImportError:
        print("Pillow未安装，正在安装...")
        try:
            import subprocess
            subprocess.check_call([sys.executable, "-m", "pip", "install", "pillow"])
            from PIL import Image
            print("✓ Pillow安装成功")
            return True
        except Exception as e:
            print(f"✗ Pillow安装失败: {e}")
            return False

def image_to_base64(image_path):
    """将图片转换为base64字符串"""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        print(f"✗ 读取图片失败: {e}")
        return None

def get_image_mime_type(image_path):
    """根据文件扩展名获取MIME类型"""
    ext = os.path.splitext(image_path)[1].lower()
    if ext == '.jpg' or ext == '.jpeg':
        return 'image/jpeg'
    elif ext == '.png':
        return 'image/png'
    elif ext == '.bmp':
        return 'image/bmp'
    elif ext == '.gif':
        return 'image/gif'
    else:
        return 'image/jpeg'  # 默认

def convert_image_to_svg(image_path, output_path=None):
    """将单个图片转换为SVG格式"""
    try:
        # 打开图片获取尺寸
        with Image.open(image_path) as img:
            width, height = img.size
        
        # 获取base64编码和MIME类型
        base64_data = image_to_base64(image_path)
        if not base64_data:
            return False
        
        mime_type = get_image_mime_type(image_path)
        
        # 确定输出路径
        if output_path is None:
            base_name = os.path.splitext(image_path)[0]
            output_path = f"{base_name}.svg"
        
        # 创建SVG内容
        svg_content = f'''<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">
  <image href="data:{mime_type};base64,{base64_data}" width="{width}" height="{height}"/>
</svg>'''
        
        # 写入SVG文件
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(svg_content)
        
        print(f"✓ 已创建: {output_path}")
        return True
        
    except Exception as e:
        print(f"✗ 转换失败 {image_path}: {e}")
        return False

def convert_directory_images(directory_path):
    """转换目录中的所有图片文件"""
    if not os.path.exists(directory_path):
        print(f"✗ 目录不存在: {directory_path}")
        return False
    
    # 支持的图片格式
    supported_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']
    
    # 获取目录中的所有图片文件
    image_files = []
    for file in os.listdir(directory_path):
        if os.path.isfile(os.path.join(directory_path, file)):
            ext = os.path.splitext(file)[1].lower()
            if ext in supported_formats:
                image_files.append(os.path.join(directory_path, file))
    
    if not image_files:
        print("✗ 未找到支持的图片文件")
        return False
    
    print(f"找到 {len(image_files)} 个图片文件")
    
    # 转换每个图片
    success_count = 0
    for image_file in image_files:
        if convert_image_to_svg(image_file):
            success_count += 1
    
    print(f"\n转换完成: {success_count}/{len(image_files)} 个文件成功")
    return success_count > 0

def main():
    """主函数"""
    print("=" * 50)
    print("图片转SVG工具")
    print("=" * 50)
    
    if not check_and_install_pillow():
        return
    
    # 转换000目录中的图片
    target_directory = "000"
    print(f"\n正在处理目录: {target_directory}")
    
    if convert_directory_images(target_directory):
        print("\n🎉 转换完成！SVG文件已保存在原目录中")
        print("\n使用说明:")
        print("- SVG文件包含嵌入的图片数据，可以直接在浏览器中打开")
        print("- 这些SVG文件是位图的封装，不是真正的矢量图形")
        print("- 如需真正的矢量转换，需要使用专业的矢量 tracing 工具")
    else:
        print("\n❌ 转换失败")

if __name__ == "__main__":
    main()