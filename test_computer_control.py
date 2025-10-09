import sys
import os
import time

sys.path.append('D:\shiyan\sub_models\H_computer_control')

from model import ComputerControlModel

if __name__ == "__main__":
    # 初始化模型
    model = ComputerControlModel()
    print('Model initialized successfully')
    
    # 1. 测试单个命令执行
    print('\n=== Test 1: Single Command Execution ===')
    try:
        result = model.execute_command({'command': 'echo Hello, World!', 'shell': True})
        print('Command executed with status:', result.get('status'))
        print('Output:', result.get('stdout', 'No output'))
    except Exception as e:
        print(f'Error during command execution: {str(e)}')
    
    # 2. 测试批量命令执行（顺序）
    print('\n=== Test 2: Batch Command Execution (Sequential) ===')
    try:
        commands = [
            {'command': 'echo Command 1', 'shell': True},
            {'command': 'echo Command 2', 'shell': True},
            {'command': 'echo Command 3', 'shell': True}
        ]
        result = model.execute_batch_commands(commands, sequential=True)
        print('Batch execution status:', result.get('status'))
        print(f"Successfully executed {result.get('success_count')}/{result.get('total_commands')} commands")
        for i, cmd_result in enumerate(result.get('results', [])):
            print(f'Command {i+1} output:', cmd_result.get('stdout', 'No output'))
    except Exception as e:
        print(f'Error during batch command execution: {str(e)}')
    
    # 3. 测试资源监控
    print('\n=== Test 3: Resource Monitoring ===')
    try:
        # 监控2秒，间隔1秒
        monitor_result = model.monitor_resources(duration=2, interval=1)
        results = monitor_result.get('results', {})
        print(f"CPU usage history points: {len(results.get('cpu_history', []))}")
        if results.get('cpu_history'):
            print(f"Last CPU usage: {results.get('cpu_history')[-1]}%")
        print(f"Memory usage history points: {len(results.get('memory_history', []))}")
        if results.get('memory_history'):
            print(f"Last memory usage: {results.get('memory_history')[-1]}%")
    except Exception as e:
        print(f'Error during resource monitoring: {str(e)}')
    
    # 4. 测试模型能力信息
    print('\n=== Test 4: Model Capabilities ===')
    try:
        capabilities = model.get_model_capabilities()
        print('Supported OS:', capabilities.get('supported_os'))
        print('Available commands:', len(capabilities.get('available_commands', [])))
        print('MCP modules:', len(capabilities.get('mcp_modules', [])))
    except Exception as e:
        print(f'Error getting model capabilities: {str(e)}')
    
    print('\nAll tests completed!')