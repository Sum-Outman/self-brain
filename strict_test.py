import requests
import json

print('=== 最严格的A Management Model能力测试 ===')

# 测试1: 复杂逻辑推理
print('1. 复杂逻辑推理测试...')
try:
    response = requests.post('http://localhost:5000/api/chat/send', json={
        'message': '如果所有的鸟都会飞，企鹅是鸟，那么企鹅会飞吗？请详细解释逻辑谬误在哪里。',
        'conversation_id': 'logic-test-001'
    })
    result = response.json()
    print(f'状态码: {response.status_code}')
    print(f'响应预览: {result.get("response", "无响应")[:300]}...')
except Exception as e:
    print(f'失败: {e}')

# 测试2: 数学计算验证
print('\n2. 数学计算验证...')
try:
    response = requests.post('http://localhost:5000/api/chat/send', json={
        'message': '计算: 如果投资10000元，年利率5%，复利计算3年后的本息和是多少？列出详细计算过程。',
        'conversation_id': 'math-test-002'
    })
    result = response.json()
    print(f'状态码: {response.status_code}')
    print(f'计算结果预览: {result.get("response", "无响应")[:300]}...')
except Exception as e:
    print(f'失败: {e}')

# 测试3: 模型管理验证
print('\n3. 模型管理功能验证...')
try:
    response = requests.post('http://localhost:5000/api/chat/send', json={
        'message': '显示当前所有模型的状态，并告诉我如何选择合适的模型处理图像识别任务。',
        'conversation_id': 'model-management-003'
    })
    result = response.json()
    print(f'状态码: {response.status_code}')
    print(f'模型状态预览: {result.get("response", "无响应")[:300]}...')
except Exception as e:
    print(f'失败: {e}')

# 测试4: 直接连接验证
print('\n4. 直接连接A Management Model验证...')
try:
    response = requests.post('http://localhost:5015/api/chat', json={
        'message': '你是谁？你的具体功能是什么？'
    })
    print(f'直接连接状态码: {response.status_code}')
    print(f'直接响应: {response.text[:300]}...')
except Exception as e:
    print(f'直接连接失败: {e}')

print('\n=== 测试完成 ===')