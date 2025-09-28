# Test script to verify import of training_manager modules
import sys
import os

# Add user site-packages directory to Python path
site_packages_path = os.path.join(os.path.expanduser('~'), 'AppData', 'Roaming', 'Python', 'Python36', 'site-packages')
if site_packages_path not in sys.path:
    sys.path.append(site_packages_path)

sys.path.append('.')

# Try importing numpy
print(f"Using Python path: {sys.path}")
print('Testing import of numpy...')
try:
    import numpy
    print(f'✓ Successfully imported numpy version: {numpy.__version__}')
except Exception as e:
    print(f'✗ Failed to import numpy: {e}')

# Try importing TrainingConfigManager
print('\nTesting import of TrainingConfigManager...')
try:
    from web_interface.training_manager import TrainingConfigManager
    print('✓ TrainingConfigManager imported successfully!')
except Exception as e:
    print(f'✗ Failed to import TrainingConfigManager: {e}')

# Try importing model architectures
print('\nTesting import of model architectures...')
try:
    from web_interface.training_manager import (
        BaseModel,
        ModelA,
        create_model,
        get_model_info,
        list_available_models
    )
    print('✓ Model architectures imported successfully!')
except Exception as e:
    print(f'✗ Failed to import model architectures: {e}')

print('\nTest completed.')