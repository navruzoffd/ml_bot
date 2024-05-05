import telebot
from config import TOKEN
from ML import generate_movies

bot = telebot.TeleBot(TOKEN)

@bot.message_handler(commands=['start'])
def send_welcome(message):
    bot.reply_to(message, """\
Привет \U0001F590, я помогу тебе найти 5 фильмов по твоему запросу.
Введи запрос \U0001F447\
""")

@bot.message_handler(func=lambda message: True)
def send_movie(message):
    try:
        movies = generate_movies(message.text)
        movies_answer = '\n'.join(movies)
        bot.reply_to(message, movies_answer)
    except Exception:
        error_message = "Не способен обработать такой запрос \U0001F622..."
        bot.reply_to(message, error_message)

bot.infinity_polling()