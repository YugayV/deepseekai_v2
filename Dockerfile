FROM python:3.10-slim

WORKDIR /app

# Install TA-Lib
RUN apt-get update && apt-get install -y wget build-essential && \
    wget https://github.com/mrjbq7/ta-lib/archive/refs/tags/TA_Lib-0.4.0.tar.gz && \
    tar -xzf TA_Lib-0.4.0.tar.gz && \
    cd ta-lib-0.4.0 && \
    ./configure --prefix=/usr && \
    make && \
    make install && \
    cd .. && \
    rm -rf TA_Lib-0.4.0.tar.gz ta-lib-0.4.0 && \
    apt-get clean

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p models data

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')" || exit 1

CMD ["python", "bot.py"]