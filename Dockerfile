FROM python:3.10-slim

WORKDIR /app

# Установка TA-Lib
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

# Создаем папки
RUN mkdir -p models data

# Переменные для Railway
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV RAILWAY=true

# Запускаем бота и дашборд через supervisord или раздельно
# Для Railway лучше запускать 2 отдельных сервиса
CMD ["python", "bot.py"]