#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from sub_models.C_audio.app import app, audio_model

def test_c_audio():
    """测试C_audio模型是否能正常初始化"""
    try:
        print("测试C_audio模型初始化...")
        # 检查模型是否成功初始化
        print(f"音频模型初始化成功: {audio_model}")
        print(f"支持的音效: {list(audio_model.special_effects.keys())}")
        print("C_audio模型测试通过")
        return True
    except Exception as e:
        print(f"C_audio模型初始化失败: {e}")
        return False

if __name__ == '__main__':
    test_c_audio()
