FROM python:3.12-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/app/.data/models \
    SENTENCE_TRANSFORMERS_HOME=/app/.data/models

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN python download_models.py

EXPOSE 8501

CMD ["streamlit", "run", "stream_streamlit_frontend.py", \
     "--server.address=0.0.0.0", "--server.port=8501"]
