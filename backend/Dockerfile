FROM python:3.10-slim-bookworm

# ---- SYSTEM DEPENDENCIES FOR OPENCV AND INSIGHTFACE ----
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsndfile1 \
    ffmpeg \
    g++ \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user
RUN useradd -m -u 1000 appuser

WORKDIR /app

COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=appuser:appuser . /app

# Create models directory with correct ownership for SpeechBrain
RUN mkdir -p /app/models/speechbrain && chown -R appuser:appuser /app/models

# Set HuggingFace cache to user-accessible location
ENV HF_HOME=/home/appuser/.cache/huggingface
ENV HF_DATASETS_CACHE=/home/appuser/.cache/huggingface/datasets
RUN mkdir -p /home/appuser/.cache/huggingface && chown -R appuser:appuser /home/appuser/.cache

# Switch to non-root user
USER appuser

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
