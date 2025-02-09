FROM python:3.9-slim

# Создание рабочей директории
WORKDIR /app

# Установка зависимостей
RUN apt-get update && apt-get install -y \
    wget \
    unzip \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Копирование файлов
COPY . .

# Установка Python-зависимостей
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Скачивание модели Vosk
RUN wget https://alphacephei.com/vosk/models/vosk-model-small-ru-0.22.zip
RUN unzip vosk-model-small-ru-0.22.zip -d model
RUN rm vosk-model-small-ru-0.22.zip

# Команда для запуска бота
CMD ["python", "main.py"]
