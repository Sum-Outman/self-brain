# 外部API模型配置与使用指南

## 概述
本指南详细介绍如何在Self Brain AGI系统中配置和使用外部API模型，包括OpenAI、Anthropic、Google等第三方AI服务。

## 配置文件位置
外部API模型的配置信息存储在系统的模型注册表文件中：
```
d:\shiyan\config\model_registry.json
```

## 配置外部API模型
在`model_registry.json`文件中，您可以添加或修改外部API模型的配置。以下是一个OpenAI模型的配置示例：

```json
"gpt-3.5-turbo": {
  "id": "gpt-3.5-turbo",
  "model_type": "language",
  "model_source": "external",
  "provider": "openai",
  "api_url": "https://api.openai.com/v1",
  "api_model": "gpt-3.5-turbo",
  "api_key": "your-api-key-here",
  "timeout": 30,
  "description": "OpenAI的GPT-3.5-Turbo模型，用于通用语言处理任务",
  "version": "1.0",
  "capabilities": [
    "text_generation",
    "conversation",
    "question_answering",
    "summarization",
    "translation"
  ]
}
```

### 配置参数说明
- **id**: 模型的唯一标识符
- **model_type**: 模型类型，如"language"、"image"等
- **model_source**: 必须设置为"external"以标识为外部API模型
- **provider**: API提供商名称，支持的提供商包括：
  - `openai`: OpenAI API
  - `anthropic`: Anthropic API
  - `google`: Google Gemini API
  - `siliconflow`: SiliconFlow API
  - `openrouter`: OpenRouter API
- **api_url**: API的基础URL
- **api_model**: 要使用的具体模型名称
- **api_key**: API访问密钥（建议通过环境变量或安全方式管理）
- **timeout**: API请求超时时间（秒）
- **description**: 模型描述信息
- **version**: 模型版本
- **capabilities**: 模型支持的功能列表

## 添加新的外部API模型
要添加新的外部API模型，请按照以下步骤操作：

1. 打开`d:\shiyan\config\model_registry.json`文件
2. 在JSON对象中添加新的模型配置，使用唯一的模型ID作为键
3. 保存文件
4. 重启Manager Model服务使配置生效

## 使用外部API模型
一旦配置完成，您可以通过Web界面或API调用使用外部模型：

### 通过Web界面使用
1. 登录Self Brain AGI系统Web界面
2. 在聊天界面或任务提交界面选择配置的外部模型
3. 发送消息或提交任务

### 通过API调用
您可以通过`/api/chat`端点调用外部模型：

```python
import requests

response = requests.post(
    "http://localhost:5015/api/chat",
    json={"message": "Hello world", "model": "gpt-3.5-turbo"},
    headers={'Content-Type': 'application/json'}
)
```

## 查看可用模型
您可以通过以下方式查看系统中可用的所有模型（包括外部API模型）：

### Web界面
在聊天界面输入以下任何命令：
- `show all models`
- `show models`
- `list all models`
- `list models`
- `display all models`
- `display models`
- `model list`
- `available models`

### API调用
```python
import requests

response = requests.get("http://localhost:5015/api/models")
models = response.json()
```

## 故障排除
如果遇到外部API调用问题，请检查以下几点：

1. **API密钥是否正确**: 确保在配置中提供了有效的API密钥
2. **API URL是否正确**: 确认配置的API基础URL格式正确
3. **网络连接**: 检查系统是否可以访问外部API服务
4. **日志信息**: 查看系统日志了解具体错误信息

## 安全注意事项
- API密钥是敏感信息，请妥善保管，避免泄露
- 建议使用环境变量或密钥管理系统存储API密钥
- 在生产环境中，考虑限制对`model_registry.json`文件的访问权限

## 测试外部API集成
系统包含一个测试脚本，用于验证模型注册表加载和外部API调用功能：

```bash
cd d:\shiyan\web_interface
python test_model_registry.py
```

该脚本会测试以下功能：
1. 从配置文件加载模型注册表
2. 检查外部模型的配置
3. 模拟外部API调用（使用测试凭据）

## 支持的外部API提供商
系统当前支持以下外部API提供商：
- OpenAI
- Anthropic
- Google Gemini
- SiliconFlow
- OpenRouter
- 其他兼容OpenAI格式的API

如需支持其他API提供商，请联系系统管理员。