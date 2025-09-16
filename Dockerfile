# 使用官方Python基础镜像
FROM python:3.10-slim

# 设置工作目录
WORKDIR /app

# 复制项目文件
COPY . .

# 安装系统依赖和Python包
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc libgl1-mesa-glx && \
    pip install --no-cache-dir -r requirements.txt && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# 暴露端口范围（5000-5010, 8000）
EXPOSE 5000-5010 8000

# 设置默认启动命令（会被docker-compose覆盖）
CMD ["python", "start_system.py"]