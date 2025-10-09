import http.server
import socketserver
import sys
import os

# 设置端口号
PORT = 5000

# 自定义请求处理器以显示更多信息
class CustomHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        # 记录请求信息
        print(f"接收到请求: {self.path} 来自 {self.client_address}")
        
        # 对于特定端点返回JSON响应
        if self.path == '/health':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            response = '{"status": "healthy", "version": "1.0"}'
            self.wfile.write(response.encode())
            return
        elif self.path == '/api/system/status':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            response = '{"status": "online", "version": "1.0", "models": []}'
            self.wfile.write(response.encode())
            return
        
        # 对于其他请求，返回默认页面
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        
        # 创建一个简单的HTML响应
        html = f"""
        <html>
        <head><title>Simple HTTP Server</title></head>
        <body>
        <h1>Simple HTTP Server Running</h1>
        <p>Port: {PORT}</p>
        <p>Path: {self.path}</p>
        <p>Available endpoints:</p>
        <ul>
            <li><a href="/health">/health</a></li>
            <li><a href="/api/system/status">/api/system/status</a></li>
        </ul>
        </body>
        </html>
        """
        
        self.wfile.write(html.encode())

def run_server():
    try:
        # 创建服务器并绑定到端口
        with socketserver.TCPServer(('', PORT), CustomHandler) as httpd:
            print(f"\n=== 简单HTTP服务器信息 ===")
            print(f"服务器启动在端口: {PORT}")
            print(f"主机地址: 0.0.0.0")
            print(f"请访问: http://localhost:{PORT}")
            print(f"可用端点: /health, /api/system/status")
            print(f"按 Ctrl+C 停止服务器")
            print(f"====================")
            
            # 启动服务器，保持运行直到被中断
            httpd.serve_forever()
    except Exception as e:
        print(f"启动服务器时发生错误: {str(e)}")
        print(f"错误类型: {type(e).__name__}")
        print(f"系统可能不允许绑定到端口 {PORT}")
        sys.exit(1)

if __name__ == "__main__":
    print(f"正在准备在端口 {PORT} 启动简单HTTP服务器...")
    run_server()