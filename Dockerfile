FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl build-essential libgomp1 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p models data

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV RAILWAY=true

EXPOSE 8080

CMD ["sh", "-c", "streamlit run dashboard.py --server.address 0.0.0.0 --server.port ${PORT:-8501} --server.headless true"]