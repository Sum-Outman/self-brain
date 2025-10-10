import os
import sys
import subprocess
import time
import threading

# 设置环境变量以支持多摄像头和外部设备
web_env = os.environ.copy()
web_env['SUPPORT_MULTI_CAMERA'] = "true"
web_env['SUPPORT_EXTERNAL_DEVICES'] = "true"
web_env['FLASK_ENV'] = "development"
web_env['LOG_LEVEL'] = "DEBUG"

# 启动Web接口
print("Starting web interface directly...")
web_interface_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web_interface")

# 检查web_interface目录是否存在
if not os.path.exists(web_interface_dir):
    print(f"Error: Web interface directory not found at {web_interface_dir}")
    sys.exit(1)

# 检查web_interface目录中的app.py文件
app_file = os.path.join(web_interface_dir, "app.py")
if not os.path.exists(app_file):
    print(f"Error: app.py not found in web interface directory")
    # 尝试查找其他可能的入口文件
    files = os.listdir(web_interface_dir)
    print(f"Files in web_interface directory: {files}")
    sys.exit(1)

# 定义一个函数来读取并显示进程的输出
def read_output(proc, stream_name):
    for line in iter(lambda: proc.stdout.readline() if stream_name == 'stdout' else proc.stderr.readline(), ''):
        if line:
            print(f"[{stream_name}] {line.strip()}")

# 直接运行Web接口
process = subprocess.Popen(
    [sys.executable, app_file],
    cwd=web_interface_dir,
    env=web_env,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    universal_newlines=True
)

print(f"Web interface started with PID: {process.pid}")
print("Waiting for the server to start...")

# 创建线程来读取stdout和stderr
stdout_thread = threading.Thread(target=read_output, args=(process, 'stdout'))
stderr_thread = threading.Thread(target=read_output, args=(process, 'stderr'))

# 设置线程为守护线程，这样当主程序退出时，这些线程也会退出
stdout_thread.daemon = True
stderr_thread.daemon = True

# 启动线程
stdout_thread.start()
stderr_thread.start()

# 等待一段时间后检查进程状态
time.sleep(5)

# 检查进程是否仍在运行
if process.poll() is not None:
    print("Web interface failed to start.")
    sys.exit(1)
else:
    print("Web interface is running. Press Ctrl+C to stop.")
    
    try:
        # 保持主程序运行，直到用户按Ctrl+C
        while True:
            if process.poll() is not None:
                print(f"Web interface stopped with exit code {process.poll()}")
                break
            time.sleep(1)
    except KeyboardInterrupt:
        print("Stopping web interface...")
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
        print("Web interface stopped.")