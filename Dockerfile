FROM python:3.9-slim

# Создание рабочей директории
WORKDIR /app

# Копирование файлов
COPY . .

# Установка зависимостей
RUN pip install --no-cache-dir -r requirements.txt

# Скачивание модели Vosk
RUN apt-get update && apt-get install -y wget
RUN wget https://alphacephei.com/vosk/models/vosk-model-small-ru-0.22.zip
RUN unzip vosk-model-small-ru-0.22.zip -d model
RUN rm vosk-model-small-ru-0.22.zip

# Команда для запуска бота
CMD ["python", "main.py"]
