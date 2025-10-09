import os
from flask import Flask, render_template

# Create a simple Flask app
app = Flask(__name__, 
           template_folder=os.path.join(os.path.dirname(__file__), 'web_interface', 'templates'),
           static_folder=os.path.join(os.path.dirname(__file__), 'web_interface', 'static'))
app.secret_key = 'test_key'

# Debug information
print(f"Current working directory: {os.getcwd()}")
print(f"Template folder: {app.template_folder}")
print(f"Static folder: {app.static_folder}")

@app.route('/')
def index():
    try:
        return render_template('index.html')
    except Exception as e:
        print(f"Error rendering template: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        return f"Error: {str(e)}", 500

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=8080)
    print("Running on http://127.0.0.1:8080/")