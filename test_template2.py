import os
import sys

# Add project root to path, similar to how it's done in app.py
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Try to import some modules from the main application to simulate the environment
try:
    # Import just a few modules to see if they affect template loading
    from training_manager.advanced_train_control import TrainingController
    from camera_manager import get_camera_manager
    from web_interface.backend.enhanced_realtime_monitor import init_enhanced_realtime_monitor
    print("Successfully imported modules from main application")
except Exception as e:
    print(f"Warning: Failed to import some modules: {str(e)}")

# Create a simple Flask app for testing template loading
from flask import Flask
app = Flask(__name__, 
           template_folder=os.path.join(os.path.dirname(__file__), 'web_interface', 'templates'),
           static_folder=os.path.join(os.path.dirname(__file__), 'web_interface', 'static'))

# Print debug information
print(f"Current working directory: {os.getcwd()}")
print(f"Flask template folder: {app.template_folder}")
print(f"Template folder exists: {os.path.exists(app.template_folder)}")
print(f"index.html exists: {os.path.exists(os.path.join(app.template_folder, 'index.html'))}")

# Try to load the template
with app.app_context():
    try:
        template = app.jinja_env.get_template('index.html')
        print("Successfully loaded index.html template!")
        # Test rendering with render_template
        from flask import render_template
        html = render_template('index.html')
        print(f"Successfully rendered template (length: {len(html)} bytes)")
    except Exception as e:
        print(f"Failed to load template: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()