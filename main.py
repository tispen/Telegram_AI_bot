import json
import os.path
import pickle
import re
import nltk
import random
import model_training as mt
from telegram import Update
from telegram.ext import Updater, CommandHandler, CallbackContext, MessageHandler, Filters
import strings as st

# поиск токена для телеграмма
BOT_TOKEN = ''
if os.path.isfile(st.BOT_TOKEN_FILENAME):
    f = open(st.BOT_TOKEN_FILENAME, "r")
    BOT_TOKEN = f.read()
    f.close()
else:
    print("Пожалуйста, создайте в папке проекта файл 'token.txt' и поместите туда токен для работы телеграм бота")

config_file = open(st.BOT_DATASET_FOR_TRAINING_FILENAME, "r")
bot_config = json.load(config_file)
config_file.close()

vectorizer = ''
model = ''


# подключение обученной модели
def connect_bot_model():
    global vectorizer, model
    # если модель не существует
    if (not os.path.isfile(st.BOT_MODEL_FILENAME)) or (not os.path.isfile(st.BOT_VECTORIZER_FILENAME)):
        print("Обученная модель не найдена, запускается обучение")
        mt.train()  # запуск модуля обучения модели
        print("Обучение модели завершено")

    # чтение векторайзера из файла
    f = open(st.BOT_VECTORIZER_FILENAME, "rb")
    vectorizer = pickle.load(f)

    # чтение модели из файла
    f = open(st.BOT_MODEL_FILENAME, "rb")
    model = pickle.load(f)


connect_bot_model()


# отбрасывает знаки препинания и приводит к нижнему регистру
def my_filter(text):
    text = text.lower()
    punctuation = r'[^\w\s]'  # регулярное выражение для того, чтобы остался только текст и пробелы
    return re.sub(punctuation, "", text)


# считает насколько тексты отличаются в процентах
def is_matching(text1, text2):
    text1 = my_filter(text1)
    text2 = my_filter(text2)
    distance = nltk.edit_distance(text1, text2)  # возвращает количество букв на которое различаются тексты
    avg_length = (len(text1) + len(text2)) / 2  # средняя длина исходных текстов
    return distance / avg_length


# отпределяет намерение по тексту
def get_intent(text):
    all_intents = bot_config["intents"]
    for name, data in all_intents.items():
        for example in data["examples"]:
            if is_matching(text, example) < 0.4:
                return name


# получить ответ
def get_answer(intent):
    responses = bot_config['intents'][intent]["responses"]
    return random.choice(responses)


# бот: по фразе выдает ответ
# print(bot("Как дела?"))
def bot(text):
    intent = get_intent(text)

    if not intent:  # если намерение не найдено
        # тут подключается модель машинного обучения
        test = vectorizer.transform([text])
        intent = model.predict(test)[0]

    if intent:  # если намерение найдено
        return get_answer(intent)
    return random.choice(bot_config['failure_phrases'])


# функция ответа при получении команды /hello
def hello(update: Update, context: CallbackContext) -> None:
    update.message.reply_text(f'Hello, {update.effective_user.first_name}')


# функция будет вызвана при получении сообщения
def bot_message(update: Update, context: CallbackContext) -> None:
    text = update.message.text  # Получение сообщения
    reply = bot(text)  # подготовка ответного сообщения
    update.message.reply_text(reply)  # отправка сообщения


# связываем созданного бота с тг

updater = Updater(BOT_TOKEN)

updater.dispatcher.add_handler(CommandHandler('hello', hello))  # конфигурация: при получении команды /hello вызвать соотвествующую функцию
updater.dispatcher.add_handler(MessageHandler(Filters.text, bot_message))  # при получении любого текстового сообщения вызывается функция botMessage

print("Бот запущен")
updater.start_polling()
updater.idle()
