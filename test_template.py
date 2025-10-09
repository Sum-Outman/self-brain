from flask import Flask, render_template
import json
from datetime import datetime
import os

# 创建Flask应用并指定模板和静态文件目录
template_dir = os.path.join(os.path.dirname(__file__), 'training_manager', 'templates')
static_dir = os.path.join(os.path.dirname(__file__), 'training_manager', 'static')
app = Flask(__name__, template_folder=template_dir, static_folder=static_dir)

@app.route('/')
def test_template():
    # Mock data for the template
    mock_data = {
        'last_updated': datetime.now().isoformat(),
        'scheduler_stats': {
            'active_tasks': 2,
            'queued_tasks': 1,
            'completed_tasks': 15,
            'failed_tasks': 1,
            'resource_limits': {
                'cpu_usage': 75,
                'memory_usage': 80
            }
        },
        'queued_count': 1,
        'cpu_usage': 65,
        'memory_usage': 70,
        'active_tasks': [
            {
                'id': 'task_123',
                'model_ids': ['A_management', 'B_language'],
                'status': 'Running',
                'progress': 60,
                'current_epoch': 6,
                'total_epochs': 10
            },
            {
                'id': 'task_456',
                'model_ids': ['D_image', 'E_video'],
                'status': 'Running',
                'progress': 35,
                'current_epoch': 7,
                'total_epochs': 20
            }
        ],
        'available_models': [
            'A_management',
            'B_language',
            'C_audio',
            'D_image',
            'E_video',
            'F_spatial',
            'G_sensor',
            'H_computer_control',
            'I_knowledge',
            'J_motion',
            'K_programming'
        ],
        'training_history': [
            {
                'task_id': 'task_789',
                'status': 'Completed',
                'timestamp': datetime.now().isoformat(),
                'models': ['B_language', 'I_knowledge']
            },
            {
                'task_id': 'task_101',
                'status': 'Failed',
                'timestamp': datetime.now().isoformat(),
                'models': ['D_image']
            }
        ]
    }
    
    return render_template('training_dashboard.html', **mock_data)

if __name__ == '__main__':
    print(f"Testing template at: {template_dir}")
    app.run(debug=True, port=5013)