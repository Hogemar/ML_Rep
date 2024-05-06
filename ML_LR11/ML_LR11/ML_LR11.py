import pandas as pd
import numpy as np
import sklearn
import re
import nltk

from nltk import tokenize
from nltk.tokenize import RegexpTokenizer

import random
from collections import Counter, defaultdict

nltk.download('punkt')
def get_text_for_label(df, label):#Функция получает тексты для заданной категории (label) из DataFrame.
    label_query = 'label == ' + str(label)
    df_label = df.query(label_query).drop(['label'], axis=1)
    return df_label.feedback.values.tolist()

def tokenize_sentences(text_corp):#Функция токенизирует предложения из списка текстов.
    token_corp = []
    tokenizer = RegexpTokenizer(r'\w+')
    for text in text_corp:
        sentences = tokenize.sent_tokenize(text)
        for sent in sentences:
            tokens = tokenizer.tokenize(sent)
            token_corp.extend(tokens)
        token_corp.append('END_SENT_START')  # В конце каждого предложения добавляем фиктивный токен
    return token_corp

def get_bigrams(token_list):#Функция создает список биграмм из списка токенов
    bigram_corp = []
    for i in range(len(token_list) - 1):
        bigram = token_list[i] + ' ' + token_list[i + 1]
        bigram_corp.append(bigram)  # Получим список биграмм
    return bigram_corp

'''
Функция генерирует тексты на основе списка токенов и биграмм.
Биграмма - это последовательность из двух слов, которые встречаются рядом друг с другом в тексте.
Она принимает параметры: метку (label), 
количество генерируемых текстов (count_text), 
количество предложений в каждом тексте (count_sent), 
количество слов в каждом предложении (count_word).
'''


'''
Генерация текстов:

Для каждого текста генерируется случайное начальное слово.
Затем для каждого предложения в тексте генерируются случайные слова на основе биграмм (пар последовательных слов) из корпуса.
Если для текущего слова не найдено подходящее следующее слово в биграммах, выбирается случайное слово из всего корпуса.
Сгенерированный текст добавляется в список текстов.
Создается DataFrame из списка сгенерированных текстов, включая метку (label).
'''
def generate_texts(token_list, bigram_list, label, count_text, count_sent, count_word):
    # Создаем словарь для подсчета биграмм "исключений"
    exceptions_bigramm = defaultdict(int)
    texts = []
    for it_text in range(count_text):  # Цикл с диапазоном кол-ва текстов
        text = ''
        for it_sent in range(count_sent):  # Цикл с диапазоном кол-ва предложений в тексте
            final_sent = ''
            # Генерируем случайное слово для начала предложения для обеспечения стохастического процесса генерации предложения
            start_word = random.choice(token_list)
            final_sent += start_word.capitalize()
            for step in range(count_word - 1):
                next_word = None
                # Формируем список биграмм, начинающихся с текущего слова
                possible_bigrams = [bigram for bigram in bigram_list if bigram.startswith(start_word)]
                if possible_bigrams:
                    # Случайно выбираем следующее слово из списка биграмм
                    next_bigram = random.choice(possible_bigrams)
                    next_word = next_bigram.split()[1]
                if next_word is None:
                    # Если не удалось найти подходящее слово, выбираем случайное слово из всего корпуса
                    next_word = random.choice(token_list)
                final_sent += ' ' + next_word
                start_word = next_word
            final_sent += '. '
            text += final_sent
        texts.append(text)
    generation_text_df = pd.DataFrame(texts, columns=['feedback'])  # Формируем фрейм из списка
    generation_text_df['label'] = label
    return generation_text_df[['label', 'feedback']]

df_fin_feedback = pd.DataFrame({
    'feedback': [
        # Объемные отзывы для 5 звезд
        "Отличный отель! Великолепный сервис, прекрасные номера и вкусная еда. Рекомендую всем!",
        "Прекрасный отель! Замечательное место для отдыха с семьей. Бассейн, детский клуб, все очень понравилось.",
        "Отель превзошел все ожидания! Уютные номера, великолепный вид, вежливый персонал. Обязательно вернемся сюда еще раз.",
        # Объемные отзывы для 4 звезд
        "Хороший отель. В целом все понравилось, но были небольшие недочеты. Номера чистые, персонал внимательный.",
        "Приятный отель. Неплохое расположение, удобные номера, но немного шумно. Завтраки вкусные.",
        "Хороший отель за свою цену. Номера комфортные, персонал дружелюбный. Но есть небольшие проблемы с чистотой.",
        # Объемные отзывы для 3 звезд
        "Средний отель. Не совсем соответствует ожиданиям. Номера чистые, но не очень уютные. Персонал отзывчивый.",
        "Обычный отель. Не особо впечатлил. Номера стандартные, завтраки обычные. Цена немного завышена.",
        "Средний отель за среднюю цену. Ничего особенного, но и не плохо. Чисто, но нет ничего выдающегося."
    ],
    'label': [5, 5, 5, 4, 4, 4, 3, 3, 3]  # Указываем соответствующие рейтинги для каждого отзыва
})

def normalize_text(text):
    # Удаление фиктивного токена
    text = text.replace('end_sent_start', '')
    # Удаление лишних символов и пробелов
    text = re.sub(r'[^\w\s]', '', text)
    # Коррекция заглавных букв
    text = text.capitalize()
    # Добавление точки в конце предложения, если ее нет
    if not text.endswith('.'):
        text += '.'
    return text

# Получаем тексты и биграммы для отзывов на 5 звезд
feedback_label_5_stars = get_text_for_label(df_fin_feedback, 5)
token_label_5_stars = tokenize_sentences(feedback_label_5_stars)
bigram_label_5_stars = get_bigrams(token_label_5_stars)

# Получаем тексты и биграммы для отзывов на 3 звезды
feedback_label_3_stars = get_text_for_label(df_fin_feedback, 3)
token_label_3_stars = tokenize_sentences(feedback_label_3_stars)
bigram_label_3_stars = get_bigrams(token_label_3_stars)

generated_texts_label_5_stars = generate_texts(token_label_5_stars, bigram_label_5_stars, label=5, count_text=3, count_sent=2, count_word=10)

generated_texts_label_5_stars['feedback'] = generated_texts_label_5_stars['feedback'].apply(normalize_text)


generated_texts_label_3_stars = generate_texts(token_label_3_stars, bigram_label_3_stars, label=3, count_text=3, count_sent=2, count_word=10)

generated_texts_label_3_stars['feedback'] = generated_texts_label_3_stars['feedback'].apply(normalize_text)


# Убираем фиктивный токен из сгенерированных текстов перед их обработкой
generated_texts_label_5_stars['feedback'] = generated_texts_label_5_stars['feedback'].str.replace(' end_sent_start', '')
generated_texts_label_3_stars['feedback'] = generated_texts_label_3_stars['feedback'].str.replace(' end_sent_start', '')
# Устанавливаем максимальную ширину столбца feedback равной None, чтобы выводить полный текст
pd.set_option('display.max_colwidth', None)
print("\n\nСгенерированные тексты для отзывов на 5 звезд после нормализации:")
print(generated_texts_label_5_stars)

print("\n\nСгенерированные тексты для отзывов на 3 звезды после нормализации:")
print(generated_texts_label_3_stars)

'''
ВЫВОД ПО ГЕНЕРАЦИИ 
Судя по результатам, тексты для отзывов на 5 звезд выглядят более позитивными что указывает на положительный опыт пребывания в отеле. 
С другой стороны, тексты для отзывов на 3 звезды выражают более сдержанную оценку и содержат слова, указывающие на недостатки, .
Таким образом, генерация текстов отражает различия в отзывах на отели с разными оценками, и можно сделать вывод, 
что модель генерации текстов учитывает контекст и генерирует соответствующие оценке отзывы.
'''
