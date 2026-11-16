# 第 8 课第 5 步：把 app.py 装进容器。
# 用法（在 ~/Desktop/day1 里，先生成依赖清单，再构建、再运行）：
#   pip freeze > requirements.txt
#   docker build -t rag-app .
#   docker run -d --name rag-app -p 8000:8000 \
#     -e DEEPSEEK_API_KEY=$DEEPSEEK_API_KEY \
#     -e SILICONFLOW_API_KEY=$SILICONFLOW_API_KEY rag-app
#   curl http://localhost:8000/stats
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# app.py 还 import 了第 1 课的 chunk_search.py；与其一个个补，
# 不如把 day1 的 .py 全拷进去——多拷的文件只是躺在镜像里，不运行
COPY *.py sample.txt ./
EXPOSE 8000
# 0.0.0.0 是关键：只监听 127.0.0.1 的话，容器外面连不进来
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
