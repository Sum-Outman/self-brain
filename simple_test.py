from flask import Flask, jsonify
import datetime

app = Flask(__name__)

@app.route('/')
def hello():
    return "Hello, World! Flask is running on port 5000!"

@app.route('/health')
def health():
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "version": "1.0"
    })

@app.route('/api/system/status')
def system_status():
    return jsonify({
        "status": "online",
        "timestamp": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "version": "1.0",
        "models": []
    })

if __name__ == '__main__':
    print("Starting simple test server on http://0.0.0.0:5000")
    app.run(host='0.0.0.0', port=5000, debug=False)