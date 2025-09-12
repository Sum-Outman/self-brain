# 子模型数据协调系统 | Sub-model Data Coordination System
# Copyright 2025 AGI System Team
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

import asyncio
import json
import logging
import hashlib
import time  # 添加time模块导入
import xml.etree.ElementTree as ET  # 添加ET模块导入
from typing import Dict, Any, Optional, List
from manager_model.emotion_engine import EmotionEngine

class DataConverter:
    """数据转换器基类 (Data Converter Base Class)"""
    def __init__(self, name: str, input_format: str, output_format: str):
        self.name = name
        self.input_format = input_format
        self.output_format = output_format
        
    def convert(self, data: Any) -> Any:
        """执行数据转换（子类需实现） (Perform data conversion - must be implemented by subclasses)"""
        raise NotImplementedError("convert方法必须在子类中实现")

class DataBroker:
    """子模型数据协调系统 (Sub-model Data Coordination System)"""
    def __init__(self, emotion_engine: EmotionEngine):
        """
        初始化数据代理
        (Initialize the data broker)
        
        :param emotion_engine: 情感引擎实例
        (Emotion engine instance)
        """
        self.emotion_engine = emotion_engine
        self.logger = logging.getLogger('DataBroker')
        
        # 数据转换器注册表 {转换器名称: 转换器实例} (Data converter registry {converter name: converter instance})
        self.converters: Dict[str, DataConverter] = {}
        
        # 数据缓存 {数据哈希: (数据, 过期时间)} (Data cache {data hash: (data, expiration time)})
        self.data_cache: Dict[str, tuple] = {}
        
        # 转换路径缓存 {(输入格式, 输出格式): 转换器链} (Conversion path cache {(input format, output format): converter chain})
        self.conversion_paths: Dict[tuple, List[str]] = {}
        
        # 数据验证器 {数据类型: 验证函数} (Data validators {data type: validation function})
        self.validators: Dict[str, callable] = {}
        
        # 错误处理策略 {错误类型: 处理函数} (Error handling strategies {error type: handler function})
        self.error_handlers: Dict[str, callable] = {}
        
        # 默认缓存时间（秒） (Default cache time (seconds))
        self.default_cache_ttl = 300
        
        # 自动注册默认验证器和错误处理器
        self._register_defaults()
        
    def _register_defaults(self):
        """注册默认验证器和错误处理器 (Register default validators and error handlers)"""
        # 此方法已符合要求，无需修改
        # 注册验证器
        self.register_validator("json", validate_json)
        self.register_validator("xml", validate_xml)
        
        # 注册错误处理器
        self.register_error_handler("INVALID_DATA", log_error_handler)
        self.register_error_handler("NO_CONVERSION_PATH", log_error_handler)
        self.register_error_handler("MISSING_CONVERTER", log_error_handler)
        self.register_error_handler("CONVERSION_FAILED", log_error_handler)
        self.register_error_handler("INVALID_OUTPUT_DATA", log_error_handler)
        
        self.logger.info("已注册默认验证器和错误处理器 (Default validators and error handlers registered)")

    def register_converter(self, converter: DataConverter):
        """
        注册数据转换器
        (Register data converter)
        """
        if converter.name in self.converters:
            self.logger.warning(f"转换器 '{converter.name}' 已存在，将被覆盖 (Converter '{converter.name}' already exists, will be overwritten)")
            
        self.converters[converter.name] = converter
        self.logger.info(f"已注册转换器: {converter.name} ({converter.input_format} -> {converter.output_format}) "
                        f"(Registered converter: {converter.name} ({converter.input_format} -> {converter.output_format}))")
        
        # 清除相关转换路径缓存 (Clear related conversion path cache)
        keys_to_remove = [k for k in self.conversion_paths 
                         if k[0] == converter.input_format or k[1] == converter.output_format]
        for key in keys_to_remove:
            del self.conversion_paths[key]

    def register_validator(self, data_type: str, validator: callable):
        """
        注册数据验证器
        (Register data validator)
        """
        self.validators[data_type] = validator
        self.logger.info(f"已注册 {data_type} 数据验证器 (Registered {data_type} data validator)")

    def register_error_handler(self, error_type: str, handler: callable):
        """
        注册错误处理策略
        (Register error handling strategy)
        """
        self.error_handlers[error_type] = handler
        self.logger.info(f"已注册 {error_type} 错误处理策略 (Registered {error_type} error handling strategy)")

    def route_data(self, source_data: Any, source_format: str, target_format: str) -> Any:
        """
        路由和转换数据
        (Route and convert data)
        
        :param source_data: 原始数据
        (Source data)
        :param source_format: 原始数据格式
        (Source data format)
        :param target_format: 目标数据格式
        (Target data format)
        :return: 转换后的数据
        (Converted data)
        """
        # 1. 数据验证 (Data validation)
        if not self._validate_data(source_data, source_format):
            # 记录更详细的验证错误信息 (Log more detailed validation error)
            data_sample = str(source_data)[:100] + ('...' if len(str(source_data)) > 100 else '')
            self.logger.error(f"源数据验证失败: {source_format} (数据样本: {data_sample}) (Source data validation failed: {source_format} (data sample: {data_sample}))")
            self._handle_error("INVALID_DATA", source_data)
            return None
            
        # 2. 检查缓存 (Check cache)
        cache_key = self._generate_cache_key(source_data, source_format, target_format)
        cached_data = self._get_cached_data(cache_key)
        if cached_data is not None:
            self.logger.debug(f"从缓存获取数据: {cache_key} (Retrieved data from cache: {cache_key})")
            return cached_data
            
        # 3. 查找转换路径 (Find conversion path)
        conversion_path = self._find_conversion_path(source_format, target_format)
        if not conversion_path:
            # 记录所有可用转换器信息以帮助诊断 (Log all available converters for diagnosis)
            available_converters = "\n".join(
                [f"- {name}: {conv.input_format} -> {conv.output_format}" 
                 for name, conv in self.converters.items()]
            )
            self.logger.error(
                f"找不到转换路径: {source_format} -> {target_format}\n"
                f"可用转换器:\n{available_converters}\n"
                f"(No conversion path found: {source_format} -> {target_format}\n"
                f"Available converters:\n{available_converters})"
            )
            self._handle_error("NO_CONVERSION_PATH", source_data)
            return None
            
        # 4. 执行转换 (Perform conversion)
        current_data = source_data
        current_format = source_format
        for converter_name in conversion_path:
            converter = self.converters.get(converter_name)
            if not converter:
                # 记录所有可用转换器信息以帮助诊断 (Log all available converters for diagnosis)
                available_converters = ", ".join(self.converters.keys())
                self.logger.error(
                    f"转换器未找到: {converter_name}\n"
                    f"可用转换器: {available_converters}\n"
                    f"(Converter not found: {converter_name}\n"
                    f"Available converters: {available_converters})"
                )
                self._handle_error("MISSING_CONVERTER", current_data)
                return None
                    
            try:
                # 记录转换前的数据大小和格式 (Record data size and format before conversion)
                original_size = len(str(current_data))
                original_format = current_format
                
                # 执行转换 (Perform conversion)
                self.logger.debug(f"应用转换器: {converter_name} ({current_format} -> {converter.output_format}) (Applying converter: {converter_name} ({current_format} -> {converter.output_format}))")
                current_data = converter.convert(current_data)
                current_format = converter.output_format
                
                # 记录情感事件 (Record emotion event)
                emotion_data = {
                    "event": "DATA_CONVERSION",
                    "converter": converter_name,
                    "source_format": original_format,  # 转换前的格式 (Format before conversion)
                    "target_format": current_format,    # 转换后的格式 (Format after conversion)
                    "data_size": original_size          # 转换前的数据大小 (Data size before conversion)
                }
                self.emotion_engine.record_emotion_event(emotion_data)
                
            except Exception as e:
                self.logger.error(f"转换失败: {converter_name}, 错误: {str(e)} (Conversion failed: {converter_name}, error: {str(e)})")
                self._handle_error("CONVERSION_FAILED", current_data, exception=e)
                return None
                
        # 5. 验证输出数据 (Validate output data)
        if not self._validate_data(current_data, target_format):
            # 记录更详细的验证错误信息 (Log more detailed validation error)
            data_sample = str(current_data)[:100] + ('...' if len(str(current_data)) > 100 else '')
            self.logger.error(f"目标数据验证失败: {target_format} (数据样本: {data_sample}) (Target data validation failed: {target_format} (data sample: {data_sample}))")
            self._handle_error("INVALID_OUTPUT_DATA", current_data)
            return None
            
        # 6. 缓存结果 (Cache result)
        self._cache_data(cache_key, current_data)
        
        return current_data

    def _find_conversion_path(self, source_format: str, target_format: str) -> List[str]:
        """
        查找从源格式到目标格式的转换路径
        (Find conversion path from source to target format)
        
        :param source_format: 源数据格式
        (Source data format)
        :param target_format: 目标数据格式
        (Target data format)
        :return: 转换器名称列表
        (List of converter names)
        """
        # 检查缓存 (Check cache)
        cache_key = (source_format, target_format)
        if cache_key in self.conversion_paths:
            return self.conversion_paths[cache_key]
            
        # 使用BFS查找最短转换路径 (Use BFS to find the shortest conversion path)
        queue = [(source_format, [])]
        visited = set()
        
        while queue:
            current_format, path = queue.pop(0)
            
            if current_format == target_format:
                # 缓存找到的路径 (Cache the found path)
                self.conversion_paths[cache_key] = path
                return path
                
            if current_format in visited:
                continue
                
            visited.add(current_format)
            
            # 查找所有可能的下游转换器 (Find all possible downstream converters)
            for converter_name, converter in self.converters.items():
                if converter.input_format == current_format:
                    new_path = path + [converter_name]
                    queue.append((converter.output_format, new_path))
                    
        return None

    def _validate_data(self, data: Any, data_format: str) -> bool:
        """
        使用注册的验证器验证数据
        (Validate data using registered validators)
        
        :param data: 要验证的数据
        (Data to validate)
        :param data_format: 数据格式
        (Data format)
        :return: 验证结果
        (Validation result)
        """
        validator = self.validators.get(data_format)
        if validator is None:
            # 没有验证器视为有效 (No validator, considered valid)
            return True
            
        try:
            return validator(data)
        except Exception as e:
            self.logger.error(f"数据验证出错: {data_format}, 错误: {str(e)} (Data validation error: {data_format}, error: {str(e)})")
            return False

    def _handle_error(self, error_type: str, data: Any, exception: Exception = None):
        """
        处理数据错误
        (Handle data errors)
        
        :param error_type: 错误类型
        (Error type)
        :param data: 相关数据
        (Related data)
        :param exception: 异常对象
        (Exception object)
        """
        # 记录情感事件 (Record emotion event)
        emotion_data = {
            "event": "DATA_ERROR",
            "error_type": error_type,
            "data_size": len(str(data)),
            "exception": str(exception) if exception else None
        }
        self.emotion_engine.record_emotion_event(emotion_data)
        
        # 应用错误处理策略 (Apply error handling strategy)
        handler = self.error_handlers.get(error_type)
        if handler:
            try:
                handler(data, exception)
            except Exception as e:
                self.logger.error(f"错误处理失败: {error_type}, 错误: {str(e)} (Error handling failed: {error_type}, error: {str(e)})")

    def _generate_cache_key(self, data: Any, source_format: str, target_format: str) -> str:
        """
        生成数据缓存键
        (Generate data cache key)
        
        :param data: 要缓存的数据
        (Data to cache)
        :param source_format: 源数据格式
        (Source data format)
        :param target_format: 目标数据格式
        (Target data format)
        :return: 缓存键字符串
        (Cache key string)
        """
        # 统一处理不同数据类型 (Handle different data types uniformly)
        try:
            if isinstance(data, (dict, list, tuple)):
                data_str = json.dumps(data, sort_keys=True)
            elif isinstance(data, str):
                data_str = data
            else:
                # 尝试序列化其他类型 (Try to serialize other types)
                try:
                    data_str = json.dumps(data)
                except TypeError:
                    data_str = str(data)
        except Exception as e:
            self.logger.error(f"缓存键生成失败: {str(e)} (Cache key generation failed)")
            # 回退到简单表示 (Fallback to simple representation)
            data_str = f"{type(data)}-{hash(data)}"
            
        key_str = f"{source_format}->{target_format}:{data_str}"
        cache_key = hashlib.sha256(key_str.encode()).hexdigest()
        
        # 记录调试信息 (Log debug information)
        if self.logger.isEnabledFor(logging.DEBUG):
            # 安全处理各种数据类型 (Safely handle various data types)
            try:
                if isinstance(data, (bytes, bytearray)):
                    data_sample = f"<binary data, length={len(data)}>"
                elif isinstance(data, str):
                    data_sample = data[:50] + ('...' if len(data) > 50 else '')
                else:
                    data_str = str(data)
                    data_sample = data_str[:50] + ('...' if len(data_str) > 50 else '')
            except Exception as e:
                data_sample = f"<data representation error: {str(e)}>"
                
            self.logger.debug(f"生成缓存键: {cache_key} (数据样本: {data_sample}) (Generated cache key: {cache_key} (data sample: {data_sample}))")
            
        return cache_key

    def _cache_data(self, key: str, data: Any, ttl: int = None):
        """缓存数据 (Cache data)"""
        ttl = ttl or self.default_cache_ttl
        expire_time = time.time() + ttl
        self.data_cache[key] = (data, expire_time)
        self.logger.debug(f"已缓存数据: {key}, TTL: {ttl}s")

    def _get_cached_data(self, key: str) -> Optional[Any]:
        """从缓存获取数据 (Get data from cache)"""
        cached = self.data_cache.get(key)
        if not cached:
            return None
            
        data, expire_time = cached
        if time.time() > expire_time:
            del self.data_cache[key]
            return None
            
        return data

    async def start(self):
        """启动数据代理（异步） (Start data broker - asynchronous)"""
        self.logger.info("数据代理已启动")
        
        # 启动后台清理任务
        # 兼容Python 3.6及以下版本
        if hasattr(asyncio, 'create_task'):
            self._cleanup_task = asyncio.create_task(self._periodic_cache_cleanup())
        else:
            self._cleanup_task = asyncio.ensure_future(self._periodic_cache_cleanup())
        self.logger.info("已启动缓存清理后台任务")
        
        return True

    async def _periodic_cache_cleanup(self):
        """定期清理过期缓存的后台任务 (Background task for periodic cache cleanup)"""
        while True:
            try:
                # 每60秒清理一次过期缓存
                await asyncio.sleep(60)
                self.cleanup_expired_cache()
            except asyncio.CancelledError:
                self.logger.info("缓存清理任务被取消")
                break
            except Exception as e:
                self.logger.error(f"缓存清理任务出错: {str(e)}")
                # 出错后等待10秒再重试
                await asyncio.sleep(10)

    async def stop(self):
        """停止数据代理（异步） (Stop data broker - asynchronous)"""
        # 停止后台清理任务
        if hasattr(self, '_cleanup_task') and self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                # 等待任务被取消（最多等待5秒）
                await asyncio.wait_for(self._cleanup_task, timeout=5)
            except asyncio.CancelledError:
                self.logger.info("缓存清理任务已成功取消")
            except asyncio.TimeoutError:
                self.logger.warning("等待缓存清理任务取消超时")
            except Exception as e:
                self.logger.error(f"停止缓存清理任务时出错: {str(e)}")
            finally:
                self._cleanup_task = None
                
        self.logger.info("数据代理已停止")
        return True

    def clear_cache(self, key: str = None):
        """
        清除缓存数据 (Clear cached data)
        :param key: 可选，指定要清除的缓存键。如果为None，则清除所有缓存 
                    (Optional, specifies the cache key to clear. If None, clears all cache)
        """
        if key:
            if key in self.data_cache:
                del self.data_cache[key]
                self.logger.info(f"已清除指定缓存: {key}")
            else:
                self.logger.warning(f"缓存键不存在: {key}")
        else:
            self.data_cache.clear()
            self.logger.info("已清除所有数据缓存")
            
    def cleanup_expired_cache(self):
        """清理所有过期缓存 (Clean up all expired cache)"""
        current_time = time.time()
        expired_keys = [key for key, (_, expire_time) in self.data_cache.items() 
                       if current_time > expire_time]
        
        for key in expired_keys:
            del self.data_cache[key]
            
        if expired_keys:
            self.logger.info(f"已清理 {len(expired_keys)} 个过期缓存项")
        else:
            self.logger.debug("未找到过期缓存项")
            
    def get_cache_stats(self) -> dict:
        """获取缓存统计信息 (Get cache statistics)"""
        current_time = time.time()
        total_items = len(self.data_cache)
        expired_items = sum(1 for _, (_, expire_time) in self.data_cache.items() 
                          if current_time > expire_time)
        active_items = total_items - expired_items
        
        return {
            "total_items": total_items,
            "active_items": active_items,
            "expired_items": expired_items,
            "default_ttl": self.default_cache_ttl
        }

# 内置转换器实现
class JsonToXmlConverter(DataConverter):
    """JSON转XML转换器 (JSON to XML Converter)"""
    def __init__(self):
        """
        初始化JSON转XML转换器
        (Initialize JSON to XML converter)
        """
        super().__init__("json_to_xml", "json", "xml")
        
    def convert(self, data: dict) -> str:
        """
        将JSON字典转换为XML字符串
        (Convert JSON dictionary to XML string)
        
        :param data: JSON格式的数据
        (Data in JSON format)
        :return: XML格式的字符串
        (XML formatted string)
        """
        def dict_to_xml(tag, data_dict):
            elem = ET.Element(tag)
            for key, val in data_dict.items():
                if isinstance(val, dict):
                    elem.append(dict_to_xml(key, val))
                elif isinstance(val, list):
                    for item in val:
                        elem.append(dict_to_xml(key, item))
                else:
                    child = ET.Element(key)
                    child.text = str(val)
                    elem.append(child)
            return elem
        
        root = dict_to_xml("root", data)
        return ET.tostring(root, encoding="unicode")

class XmlToJsonConverter(DataConverter):
    """XML转JSON转换器 (XML to JSON Converter)"""
    def __init__(self):
        """
        初始化XML转JSON转换器
        (Initialize XML to JSON converter)
        """
        super().__init__("xml_to_json", "xml", "json")
        
    def convert(self, data: str) -> dict:
        """
        将XML字符串转换为JSON字典
        (Convert XML string to JSON dictionary)
        
        :param data: XML格式的字符串
        (XML formatted string)
        :return: JSON格式的数据
        (Data in JSON format)
        """
        root = ET.fromstring(data)
        
        def xml_to_dict(element):
            result = {}
            for child in element:
                if len(child) > 0:
                    result[child.tag] = xml_to_dict(child)
                else:
                    result[child.tag] = child.text
            return result
        
        return {root.tag: xml_to_dict(root)}

class TextToJsonConverter(DataConverter):
    """文本转JSON转换器（简单实现） (Text to JSON Converter - simple implementation)"""
    def __init__(self):
        """
        初始化文本转JSON转换器
        (Initialize text to JSON converter)
        """
        super().__init__("text_to_json", "text", "json")
        
    def convert(self, data: str) -> dict:
        """
        将文本转换为JSON对象
        (Convert text to JSON object)
        
        :param data: 文本数据
        (Text data)
        :return: JSON格式的数据
        (Data in JSON format)
        """
        # 简单实现：按行分割
        return {"lines": data.splitlines()}

# 内置转换器注册
def register_default_converters(broker: DataBroker):
    """
    注册默认数据转换器
    (Register default data converters)
    
    :param broker: 数据代理实例
    (Data broker instance)
    """
    broker.register_converter(JsonToXmlConverter())
    broker.register_converter(XmlToJsonConverter())
    broker.register_converter(TextToJsonConverter())
    
# 内置验证器
def validate_json(data):
    """
    验证JSON数据
    (Validate JSON data)
    
    :param data: 要验证的数据
    (Data to validate)
    :return: 验证结果
    (Validation result)
    :raises ValueError: 如果数据不是字典
    (If data is not a dictionary)
    """
    if not isinstance(data, dict):
        raise ValueError("JSON数据必须是字典 (JSON data must be a dictionary)")
    return True

def validate_xml(data):
    """
    验证XML数据
    (Validate XML data)
    
    :param data: 要验证的XML字符串
    (XML string to validate)
    :return: 验证结果
    (Validation result)
    """
    try:
        ET.fromstring(data)
        return True
    except ET.ParseError:
        return False

# 内置错误处理器
def log_error_handler(data, exception):
    """
    日志记录错误处理器
    (Log error handler)
    
    :param data: 相关数据
    (Related data)
    :param exception: 异常对象
    (Exception object)
    """
    logging.error(f"数据处理错误: {str(exception)} (Data processing error: {str(exception)})")

# 内置验证器和错误处理器注册
def register_default_validators_and_handlers(broker: DataBroker):
    """
    注册默认验证器和错误处理器
    (Register default validators and error handlers)
    
    :param broker: 数据代理实例
    (Data broker instance)
    """
    # 此方法已符合要求，无需修改
    broker.register_validator("json", validate_json)
    broker.register_validator("xml", validate_xml)
    broker.register_error_handler("INVALID_DATA", log_error_handler)
    broker.register_error_handler("NO_CONVERSION_PATH", log_error_handler)
    broker.register_error_handler("MISSING_CONVERTER", log_error_handler)
    broker.register_error_handler("CONVERSION_FAILED", log_error_handler)
    broker.register_error_handler("INVALID_OUTPUT_DATA", log_error_handler)
    logging.info("已注册默认验证器和错误处理器 (Default validators and error handlers registered)")
