FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y libgl1 libglib2.0-0 libsm6 libxext6 && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
EXPOSE 5000
CMD gunicorn app:app --bind 0.0.0.0:$PORT --workers 1 --threads 2 --timeout 120 --preload
