# 1. 使用官方 Python 3.12 完整版 (基于 Debian Bookworm)
# 体积较大，但包含了大部分标准库，最不容易出缺库的问题
FROM python:3.12

# 2. 设置环境变量
# 确保 Python 输出直接打印到 Cloud Run 日志，不被缓存
ENV PYTHONUNBUFFERED 1
# 防止 Python 生成 .pyc 字节码文件，减小体积
ENV PYTHONDONTWRITEBYTECODE 1

# 设置工作目录
ENV APP_HOME /app
WORKDIR $APP_HOME

# 3. [关键优化] 利用 Docker 缓存层机制
# 先拷贝依赖文件，再安装依赖。
# 这样如果你只改了代码（main.py）没改依赖，构建时会直接跳过 pip install，速度极快。
COPY requirements.txt .

# 4. 安装系统级依赖
# python:3.12 完整版通常自带 gcc，但为了保险起见，以及为了 psycopg2 (Postgres) 支持，安装 libpq-dev
# 注：额外安装 fonts-noto-cjk 解决 matplotlib 在容器内渲染中文变“豆腐块”的问题
RUN apt-get update && apt-get install -y \
    libpq-dev \
    gcc \
    fonts-noto-cjk \
    && rm -rf /var/lib/apt/lists/*

# 5. 安装 Python 依赖
# 升级 pip 以防止兼容性问题，并安装 setuptools (Python 3.12 移除了 distutils，某些旧包安装需要它)
RUN pip install --upgrade pip setuptools && \
    pip install --no-cache-dir -r requirements.txt

# 6. 最后再拷贝代码
# 这一步放在最后，确保代码变动不会导致重新安装依赖
COPY . ./

# Cloud Run 运行时会自动注入 PORT 环境变量（默认8080）
ENV PORT 8080

# 7. 启动命令
# 显式添加 --workers 1 (Cloud Run 容器会横向扩容，单容器单进程通常最稳定)
# 使用 shell 格式 (不带方括号) 确保 ${PORT} 变量能被正确解析
CMD exec uvicorn main:app --host 0.0.0.0 --port ${PORT} --workers 1