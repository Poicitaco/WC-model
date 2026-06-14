FROM python:3.10-slim

WORKDIR /app

# Khắc phục lỗi build liên quan đến thiếu thư viện hệ thống
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt scikit-learn==1.6.1 xgboost gunicorn

COPY . .

# Hugging Face tự động gán cổng qua biến môi trường PORT (thường là 7860)
ENV PORT=7860
ENV INFERENCE_API_URL=""
EXPOSE 7860

# Chạy bằng Gunicorn cho hiệu năng ổn định trên production
CMD ["gunicorn", "-b", "0.0.0.0:7860", "--workers", "2", "--threads", "4", "app:app"]
