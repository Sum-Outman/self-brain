import sys
import os

sys.path.append('D:\shiyan\sub_models\H_computer_control')

from model import ComputerControlModel

if __name__ == "__main__":
    # 初始化模型
    model = ComputerControlModel()
    print('Model initialized successfully')
    
    # 测试命令执行
    try:
        result = model.execute_command({'command': 'echo Hello, World!' if os.name != 'nt' else 'echo Hello, World!', 'shell': True})
        print('Command executed with status:', result.get('status'))
        print('Output:', result.get('stdout', 'No output'))
        print('Error message:', result.get('message', 'No error message'))
        print('Command:', result.get('command', 'No command'))
    except Exception as e:
        print(f'Error during command execution: {str(e)}')