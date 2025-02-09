import os
import logging
from aiogram import Bot, Dispatcher, F
from aiogram.types import Message
from aiogram.filters import Command
from aiogram.utils.keyboard import InlineKeyboardBuilder
from pydub import AudioSegment
from vosk import Model, KaldiRecognizer
from transformers import pipeline

# Настройка логирования
logging.basicConfig(level=logging.INFO)

# Токен бота (замените на ваш токен)
BOT_TOKEN = os.getenv("BOT_TOKEN")

# Инициализация бота и диспетчера
bot = Bot(token=BOT_TOKEN)
dp = Dispatcher()

# Загрузка языковой модели для анализа текста
nlp = pipeline("text-classification", model="DeepPavlov/rubert-base-cased")

# Функция для распознавания речи с использованием Vosk
def recognize_speech(file_path):
    model = Model("model")  # Убедитесь, что модель скачана в папку "model"
    recognizer = KaldiRecognizer(model, 16000)
    audio = AudioSegment.from_file(file_path)
    audio = audio.set_frame_rate(16000).set_channels(1)
    audio.export(file_path, format="wav")
    with open(file_path, "rb") as f:
        data = f.read()
        if recognizer.AcceptWaveform(data):
            result = recognizer.Result()
            return eval(result)["text"]
    return ""

# Функция для проверки трезвости
def is_sober(text: str) -> bool:
    result = nlp(text)
    label = result[0]['label']
    score = result[0]['score']
    return label != "INTOXICATED" and score < 0.5

# Обработчик команды /start
@dp.message(Command("start"))
async def start(message: Message):
    await message.answer("Привет! Я слежу за порядком в этом чате.")

# Обработчик текстовых сообщений
@dp.message(F.text)
async def handle_text_message(message: Message):
    text = message.text
    user_id = message.from_user.id
    logging.info(f"User {user_id} said: {text}")
    if not is_sober(text):
        await message.reply(f"Пользователь @{message.from_user.username}, пожалуйста, пройдите проверку на трезвость.")

# Обработчик голосовых сообщений
@dp.message(F.voice)
async def handle_voice_message(message: Message):
    voice = message.voice
    user_id = message.from_user.id
    file_id = voice.file_id
    file = await bot.get_file(file_id)
    file_path = f"voice_message_{user_id}.ogg"
    await file.download(destination=file_path)

    # Конвертация OGG в WAV
    audio = AudioSegment.from_ogg(file_path)
    audio.export(f"voice_message_{user_id}.wav", format="wav")

    # Распознавание речи
    text = recognize_speech(f"voice_message_{user_id}.wav")
    logging.info(f"User {user_id} said: {text}")
    if not is_sober(text):
        await message.reply(f"Пользователь @{message.from_user.username}, пожалуйста, пройдите проверку на трезвость.")

    # Удаление временных файлов
    os.remove(file_path)
    os.remove(f"voice_message_{user_id}.wav")

# Запуск бота
if __name__ == "__main__":
    dp.run_polling(bot)
