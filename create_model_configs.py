import os
import json
import datetime

# List of models to create configs for
models = [
    'B_language', 
    'C_audio', 
    'D_image', 
    'E_video', 
    'F_spatial', 
    'G_sensor', 
    'H_computer_control', 
    'I_knowledge', 
    'J_motion'
]

# Base directory
base_dir = os.path.join('d:', 'shiyan', 'web_interface', 'models')

# Create config for each model
for model in models:
    # Create model directory if it doesn't exist
    model_dir = os.path.join(base_dir, model)
    os.makedirs(model_dir, exist_ok=True)
    
    # Create config data
    config_data = {
        'description': f'Configuration for {model}',
        'model_source': 'local',
        'last_updated': datetime.datetime.now().isoformat()
    }
    
    # Write config file
    config_file = os.path.join(model_dir, 'config.json')
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(config_data, f, indent=2, ensure_ascii=False)
    
    print(f"Created config for model: {model}")

print("All model configurations created successfully!")