import asyncio
import json
import logging
import os
import threading
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import librosa
import soundfile as sf
from flask import Flask, jsonify, request, send_file
import speech_recognition as sr
import pyttsx3
from pydub import AudioSegment
import pyaudio
import wave

class SpeechRecognitionEngine:
    """语音识别引擎"""
    
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        
        # 配置识别器
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=1)
        
        self.recognizer.energy_threshold = 4000
        self.recognizer.dynamic_energy_threshold = True
        
        # 支持的语言
        self.supported_languages = {
            'zh-CN': '中文（简体）',
            'zh-TW': '中文（繁体）',
            'en-US': '英语（美国）',
            'en-GB': '英语（英国）',
            'ja-JP': '日语',
            'ko-KR': '韩语',
            'fr-FR': '法语',
            'de-DE': '德语',
            'es-ES': '西班牙语',
            'ru-RU': '俄语'
        }
    
    def recognize_from_microphone(self, language='zh-CN', duration=5) -> Dict:
        """从麦克风识别语音"""
        try:
            with self.microphone as source:
                print("正在聆听...")
                audio = self.recognizer.listen(source, timeout=1, phrase_time_limit=duration)
            
            text = self.recognizer.recognize_google(audio, language=language)
            
            return {
                'success': True,
                'text': text,
                'language': language,
                'confidence': 0.8,  # 简化版本
                'timestamp': datetime.now().isoformat()
            }
        
        except sr.UnknownValueError:
            return {
                'success': False,
                'error': '无法识别语音内容',
                'timestamp': datetime.now().isoformat()
            }
        except sr.RequestError as e:
            return {
                'success': False,
                'error': f'语音识别服务错误: {str(e)}',
                'timestamp': datetime.now().isoformat()
            }
    
    def recognize_from_file(self, audio_file_path: str, language='zh-CN') -> Dict:
        """从音频文件识别语音"""
        try:
            with sr.AudioFile(audio_file_path) as source:
                audio = self.recognizer.record(source)
            
            text = self.recognizer.recognize_google(audio, language=language)
            
            return {
                'success': True,
                'text': text,
                'language': language,
                'confidence': 0.8,
                'timestamp': datetime.now().isoformat()
            }
        
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

class ToneAnalyzer:
    """语调分析器"""
    
    def __init__(self):
        self.emotion_tones = {
            'happy': {'pitch_mean': 200, 'pitch_var': 50, 'speed': 1.2},
            'sad': {'pitch_mean': 150, 'pitch_var': 20, 'speed': 0.8},
            'angry': {'pitch_mean': 180, 'pitch_var': 80, 'speed': 1.5},
            'fear': {'pitch_mean': 220, 'pitch_var': 60, 'speed': 1.3},
            'surprise': {'pitch_mean': 250, 'pitch_var': 70, 'speed': 1.4},
            'neutral': {'pitch_mean': 170, 'pitch_var': 30, 'speed': 1.0}
        }
    
    def analyze_tone(self, audio_data: np.ndarray, sr_rate: int) -> Dict:
        """分析音频语调特征"""
        try:
            # 提取基频
            pitches, magnitudes = librosa.piptrack(y=audio_data, sr=sr_rate)
            
            pitch_values = []
            for t in range(pitches.shape[1]):
                index = magnitudes[:, t].argmax()
                pitch = pitches[index, t]
                if pitch > 0:
                    pitch_values.append(pitch)
            
            if not pitch_values:
                return {'tone': 'neutral', 'confidence': 0.5}
            
            pitch_mean = np.mean(pitch_values)
            pitch_var = np.std(pitch_values)
            
            # 计算语速
            tempo, _ = librosa.beat.beat_track(y=audio_data, sr=sr_rate)
            speed = tempo / 120.0  # 标准化
            
            # 匹配情感语调
            best_match = 'neutral'
            min_distance = float('inf')
            
            for emotion, features in self.emotion_tones.items():
                distance = abs(pitch_mean - features['pitch_mean']) + \
                          abs(pitch_var - features['pitch_var']) + \
                          abs(speed - features['speed'])
                
                if distance < min_distance:
                    min_distance = distance
                    best_match = emotion
            
            return {
                'tone': best_match,
                'confidence': max(0, 1 - min_distance / 100),
                'features': {
                    'pitch_mean': float(pitch_mean),
                    'pitch_variation': float(pitch_var),
                    'speed': float(speed)
                },
                'timestamp': datetime.now().isoformat()
            }
        
        except Exception as e:
            return {'tone': 'neutral', 'confidence': 0.5, 'error': str(e)}

class SpeechSynthesizer:
    """语音合成器"""
    
    def __init__(self):
        self.engine = pyttsx3.init()
        
        # 配置语音属性
        self.voices = self.engine.getProperty('voices')
        self.engine.setProperty('rate', 150)
        self.engine.setProperty('volume', 0.9)
        
        # 可用的声音
        self.available_voices = {}
        for i, voice in enumerate(self.voices):
            self.available_voices[f'voice_{i}'] = {
                'name': voice.name,
                'id': voice.id,
                'languages': voice.languages if hasattr(voice, 'languages') else ['zh-CN']
            }
    
    def synthesize_speech(self, text: str, voice_id='voice_0', speed=150, volume=0.9) -> str:
        """合成语音"""
        try:
            # 设置语音属性
            self.engine.setProperty('rate', speed)
            self.engine.setProperty('volume', volume)
            
            if voice_id in self.available_voices:
                voice_info = self.available_voices[voice_id]
                for voice in self.voices:
                    if voice.id == voice_info['id']:
                        self.engine.setProperty('voice', voice.id)
                        break
            
            # 生成音频文件
            output_file = f"temp_speech_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
            self.engine.save_to_file(text, output_file)
            self.engine.runAndWait()
            
            return output_file
        
        except Exception as e:
            return f"合成失败: {str(e)}"
    
    def get_available_voices(self) -> Dict:
        """获取可用的语音"""
        return self.available_voices

class MusicGenerator:
    """音乐生成器"""
    
    def __init__(self):
        self.scales = {
            'major': [0, 2, 4, 5, 7, 9, 11],
            'minor': [0, 2, 3, 5, 7, 8, 10],
            'pentatonic': [0, 2, 4, 7, 9]
        }
        
        self.instruments = {
            'piano': {'sample_rate': 44100, 'harmonics': [1, 0.5, 0.25, 0.125]},
            'guitar': {'sample_rate': 22050, 'harmonics': [1, 0.7, 0.5, 0.3]},
            'violin': {'sample_rate': 44100, 'harmonics': [1, 0.8, 0.6, 0.4]}
        }
    
    def generate_melody(self, key='C', scale='major', tempo=120, duration=10) -> np.ndarray:
        """生成旋律"""
        try:
            sample_rate = 22050
            t = np.linspace(0, duration, int(sample_rate * duration))
            
            # 生成音符序列
            scale_notes = self.scales.get(scale, self.scales['major'])
            melody = np.zeros_like(t)
            
            note_duration = 60 / tempo  # 每拍时长
            
            for i in range(0, len(t), int(sample_rate * note_duration)):
                note_index = i // int(sample_rate * note_duration)
                note = scale_notes[note_index % len(scale_notes)]
                
                # 计算频率
                freq = 440 * (2 ** ((note + {'C': -9, 'D': -7, 'E': -5, 'F': -4, 'G': -2, 'A': 0, 'B': 2}[key]) / 12))
                
                # 生成正弦波
                note_t = t[i:i + int(sample_rate * note_duration)]
                if len(note_t) > 0:
                    note_wave = np.sin(2 * np.pi * freq * note_t)
                    melody[i:i + len(note_t)] = note_wave
            
            return melody
        
        except Exception as e:
            return np.zeros(1000)
    
    def generate_chord_progression(self, key='C', progression=['I', 'IV', 'V', 'I'], tempo=120) -> np.ndarray:
        """生成和弦进行"""
        try:
            sample_rate = 22050
            duration_per_chord = 2  # 每个和弦2秒
            total_duration = len(progression) * duration_per_chord
            
            t = np.linspace(0, total_duration, int(sample_rate * total_duration))
            chord_audio = np.zeros_like(t)
            
            chord_freqs = {
                'C': [261.63, 329.63, 392.00],  # C-E-G
                'D': [293.66, 369.99, 440.00],  # D-F#-A
                'E': [329.63, 415.30, 493.88],  # E-G#-B
                'F': [349.23, 440.00, 523.25],  # F-A-C
                'G': [392.00, 493.88, 587.33],  # G-B-D
                'A': [440.00, 523.25, 659.25],  # A-C#-E
                'B': [493.88, 587.33, 739.99]   # B-D#-F#
            }
            
            for i, chord_name in enumerate(progression):
                start_idx = i * int(sample_rate * duration_per_chord)
                end_idx = (i + 1) * int(sample_rate * duration_per_chord)
                
                if chord_name == 'I':
                    chord = key
                elif chord_name == 'IV':
                    chord = {'C': 'F', 'D': 'G', 'E': 'A', 'F': 'Bb', 'G': 'C', 'A': 'D', 'B': 'E'}[key]
                elif chord_name == 'V':
                    chord = {'C': 'G', 'D': 'A', 'E': 'B', 'F': 'C', 'G': 'D', 'A': 'E', 'B': 'F#'}[key]
                else:
                    chord = key
                
                freqs = chord_freqs.get(chord, chord_freqs['C'])
                
                chord_t = t[start_idx:end_idx]
                if len(chord_t) > 0:
                    chord_wave = np.zeros_like(chord_t)
                    for freq in freqs:
                        chord_wave += 0.3 * np.sin(2 * np.pi * freq * chord_t)
                    chord_audio[start_idx:end_idx] = chord_wave
            
            return chord_audio
        
        except Exception as e:
            return np.zeros(1000)

class NoiseProcessor:
    """噪音处理器"""
    
    def __init__(self):
        self.noise_types = {
            'white': {'description': '白噪音'},
            'pink': {'description': '粉噪音'},
            'brown': {'description': '棕噪音'},
            'blue': {'description': '蓝噪音'}
        }
    
    def generate_noise(self, noise_type='white', duration=5, amplitude=0.1) -> np.ndarray:
        """生成噪音"""
        try:
            sample_rate = 44100
            samples = int(sample_rate * duration)
            
            if noise_type == 'white':
                noise = np.random.normal(0, amplitude, samples)
            elif noise_type == 'pink':
                # 粉噪音生成
                noise = np.random.normal(0, amplitude, samples)
                # 应用1/f滤波器（简化版本）
                noise = np.convolve(noise, [1, 0.5, 0.25, 0.125], mode='same')
            elif noise_type == 'brown':
                # 棕噪音生成
                noise = np.cumsum(np.random.normal(0, amplitude, samples))
                noise = noise / np.max(np.abs(noise)) * amplitude
            elif noise_type == 'blue':
                # 蓝噪音生成
                noise = np.random.normal(0, amplitude, samples)
                # 应用高通滤波器（简化版本）
                noise = np.diff(noise, prepend=0)
            else:
                noise = np.random.normal(0, amplitude, samples)
            
            return noise
        
        except Exception as e:
            return np.zeros(samples)
    
    def remove_noise(self, audio_data: np.ndarray, noise_gate=-40) -> np.ndarray:
        """噪音消除"""
        try:
            # 简单的噪音门限处理
            threshold = 10 ** (noise_gate / 20)
            
            # 计算音频能量
            frame_length = 1024
            hop_length = 512
            
            energy = np.array([
                np.sum(audio_data[i:i+frame_length]**2)
                for i in range(0, len(audio_data)-frame_length, hop_length)
            ])
            
            # 应用噪音门限
            mask = energy > threshold
            
            # 扩展掩码到原始音频长度
            expanded_mask = np.repeat(mask, hop_length)[:len(audio_data)]
            
            # 应用掩码
            cleaned_audio = audio_data * expanded_mask
            
            return cleaned_audio
        
        except Exception as e:
            return audio_data

class CAudioModel:
    """C音频处理模型 - 完整的音频处理能力"""
    
    def __init__(self):
        self.app = Flask(__name__)
        self.setup_routes()
        
        # 初始化各个处理器
        self.speech_recognition = SpeechRecognitionEngine()
        self.tone_analyzer = ToneAnalyzer()
        self.speech_synthesizer = SpeechSynthesizer()
        self.music_generator = MusicGenerator()
        self.noise_processor = NoiseProcessor()
        
        # 模型状态
        self.model_status = {
            'status': 'ready',
            'last_activity': datetime.now().isoformat(),
            'processed_requests': 0,
            'features': [
                'speech_recognition',
                'tone_analysis',
                'speech_synthesis',
                'music_generation',
                'noise_processing',
                'multilingual_support',
                'real_time_processing'
            ],
            'supported_formats': ['wav', 'mp3', 'flac', 'aac', 'ogg'],
            'sample_rates': [8000, 16000, 22050, 44100, 48000]
        }
        
        # 音频缓存
        self.audio_cache = {}
        
    def setup_routes(self):
        """设置API路由"""
        
        @self.app.route('/health', methods=['GET'])
        def health_check():
            return jsonify({
                'status': 'healthy',
                'model': 'C_Audio',
                'timestamp': datetime.now().isoformat()
            })
        
        @self.app.route('/api/speech/recognize', methods=['POST'])
        def recognize_speech():
            try:
                audio_file = request.files.get('audio')
                language = request.form.get('language', 'zh-CN')
                source = request.form.get('source', 'file')  # file or microphone
                
                if source == 'microphone':
                    result = self.speech_recognition.recognize_from_microphone(language)
                elif audio_file:
                    # 保存上传的文件
                    filename = f"temp_audio_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
                    audio_file.save(filename)
                    result = self.speech_recognition.recognize_from_file(filename, language)
                    
                    # 清理临时文件
                    if os.path.exists(filename):
                        os.remove(filename)
                else:
                    result = {'success': False, 'error': '未提供音频文件'}
                
                self.model_status['processed_requests'] += 1
                self.model_status['last_activity'] = datetime.now().isoformat()
                
                return jsonify(result)
            
            except Exception as e:
                return jsonify({'success': False, 'error': str(e)}), 500
        
        @self.app.route('/api/speech/synthesize', methods=['POST'])
        def synthesize_speech():
            try:
                data = request.json
                text = data.get('text', '')
                voice_id = data.get('voice_id', 'voice_0')
                speed = data.get('speed', 150)
                volume = data.get('volume', 0.9)
                
                output_file = self.speech_synthesizer.synthesize_speech(text, voice_id, speed, volume)
                
                if os.path.exists(output_file):
                    return send_file(output_file, as_attachment=True, download_name='speech.wav')
                else:
                    return jsonify({'success': False, 'error': '语音合成失败'}), 500
            
            except Exception as e:
                return jsonify({'success': False, 'error': str(e)}), 500
        
        @self.app.route('/api/tone/analyze', methods=['POST'])
        def analyze_tone():
            try:
                audio_file = request.files.get('audio')
                
                if not audio_file:
                    return jsonify({'success': False, 'error': '未提供音频文件'}), 400
                
                # 保存上传的文件
                filename = f"temp_tone_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
                audio_file.save(filename)
                
                # 加载音频
                audio_data, sr_rate = librosa.load(filename)
                
                # 分析语调
                result = self.tone_analyzer.analyze_tone(audio_data, sr_rate)
                
                # 清理临时文件
                if os.path.exists(filename):
                    os.remove(filename)
                
                self.model_status['processed_requests'] += 1
                self.model_status['last_activity'] = datetime.now().isoformat()
                
                return jsonify(result)
            
            except Exception as e:
                return jsonify({'success': False, 'error': str(e)}), 500
        
        @self.app.route('/api/music/generate', methods=['POST'])
        def generate_music():
            try:
                data = request.json
                key = data.get('key', 'C')
                scale = data.get('scale', 'major')
                tempo = data.get('tempo', 120)
                duration = data.get('duration', 10)
                type = data.get('type', 'melody')  # melody or chord
                
                if type == 'melody':
                    music = self.music_generator.generate_melody(key, scale, tempo, duration)
                elif type == 'chord':
                    progression = data.get('progression', ['I', 'IV', 'V', 'I'])
                    music = self.music_generator.generate_chord_progression(key, progression, tempo)
                else:
                    music = self.music_generator.generate_melody(key, scale, tempo, duration)
                
                # 保存为音频文件
                output_file = f"temp_music_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
                sf.write(output_file, music, 22050)
                
                if os.path.exists(output_file):
                    return send_file(output_file, as_attachment=True, download_name='music.wav')
                else:
                    return jsonify({'success': False, 'error': '音乐生成失败'}), 500
            
            except Exception as e:
                return jsonify({'success': False, 'error': str(e)}), 500
        
        @self.app.route('/api/noise/generate', methods=['POST'])
        def generate_noise():
            try:
                data = request.json
                noise_type = data.get('type', 'white')
                duration = data.get('duration', 5)
                amplitude = data.get('amplitude', 0.1)
                
                noise = self.noise_processor.generate_noise(noise_type, duration, amplitude)
                
                # 保存为音频文件
                output_file = f"temp_noise_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
                sf.write(output_file, noise, 44100)
                
                if os.path.exists(output_file):
                    return send_file(output_file, as_attachment=True, download_name='noise.wav')
                else:
                    return jsonify({'success': False, 'error': '噪音生成失败'}), 500
            
            except Exception as e:
                return jsonify({'success': False, 'error': str(e)}), 500
        
        @self.app.route('/api/noise/remove', methods=['POST'])
        def remove_noise():
            try:
                audio_file = request.files.get('audio')
                noise_gate = float(request.form.get('noise_gate', -40))
                
                if not audio_file:
                    return jsonify({'success': False, 'error': '未提供音频文件'}), 400
                
                # 保存上传的文件
                filename = f"temp_clean_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
                audio_file.save(filename)
                
                # 加载音频
                audio_data, sr_rate = librosa.load(filename)
                
                # 噪音消除
                cleaned_audio = self.noise_processor.remove_noise(audio_data, noise_gate)
                
                # 保存清理后的音频
                output_file = f"cleaned_{filename}"
                sf.write(output_file, cleaned_audio, sr_rate)
                
                # 清理临时文件
                if os.path.exists(filename):
                    os.remove(filename)
                
                if os.path.exists(output_file):
                    return send_file(output_file, as_attachment=True, download_name='cleaned_audio.wav')
                else:
                    return jsonify({'success': False, 'error': '噪音消除失败'}), 500
            
            except Exception as e:
                return jsonify({'success': False, 'error': str(e)}), 500
        
        @self.app.route('/api/status', methods=['GET'])
        def get_status():
            return jsonify(self.model_status)
        
        @self.app.route('/api/voices', methods=['GET'])
        def get_voices():
            return jsonify(self.speech_synthesizer.get_available_voices())
        
        @self.app.route('/api/languages', methods=['GET'])
        def get_languages():
            return jsonify(self.speech_recognition.supported_languages)
    
    def run(self, host='0.0.0.0', port=5017, debug=False):
        """运行服务"""
        logging.info(f"Starting C Audio Model on {host}:{port}")
        self.app.run(host=host, port=port, debug=debug)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    model = CAudioModel()
    model.run()