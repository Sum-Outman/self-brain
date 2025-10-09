import http.client
import sys

try:
    # 创建HTTP连接到localhost:5000
    conn = http.client.HTTPConnection('localhost', 5000, timeout=5)
    # 发送GET请求
    conn.request('GET', '/')
    # 获取响应
    response = conn.getresponse()
    
    print('Web Interface Status:', response.status)
    print('Response Headers:', response.getheaders())
    
    # 读取部分响应内容
    content = response.read(1000)  # 读取前1000个字符
    print('Response Content (partial):', content.decode('utf-8', errors='ignore'))
    
    # 根据状态码判断
    if response.status == 200:
        print('SUCCESS: Web interface is running properly!')
        sys.exit(0)
    else:
        print('WARNING: Web interface returned non-200 status code.')
        sys.exit(1)

except Exception as e:
    print('ERROR: Failed to connect to web interface:', str(e))
    sys.exit(1)
finally:
    conn.close()