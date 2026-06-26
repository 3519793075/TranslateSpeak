# —— 中文口播翻译+语音合成 容器化部署 ——
# 基于 python:3.10-slim 轻量镜像构建

FROM python:3.10-slim

# 设置工作目录
WORKDIR /app

# 安装系统依赖（如需要）
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 复制并安装 Python 依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制应用代码
COPY app.py .
COPY .env.example .env.example

# 暴露 Streamlit 默认端口
EXPOSE 8501

# 健康检查
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# 启动命令：绑定所有网络接口以支持外部访问
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
