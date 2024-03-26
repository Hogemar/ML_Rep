import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
import requests
from zipfile import ZipFile
from io import BytesIO

# Ссылка на набор данных "Spam SMS Collection" на UCI Machine Learning Repository
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"

# Загрузка и распаковка архива с данными
response = requests.get(url)
with ZipFile(BytesIO(response.content)) as zip_file:
    zip_file.extractall()

# Чтение данных из файла
with open("SMSSpamCollection", "r", encoding="utf-8") as file:
    data = file.readlines()

# Разделение текстов и меток
X_data = [line.strip().split("\t")[1] for line in data]
y_data = np.array([1 if line.strip().split("\t")[0] == "spam" else 0 for line in data])



# Создание класса SpamAnalyzer
class SpamAnalyzer:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.model = None

    def train(self, X_train, y_train):
        self.model = make_pipeline(TfidfVectorizer(), MultinomialNB(alpha=self.alpha))
        self.model.fit(X_train, y_train)

    def classify(self, email):
        return self.model.predict([email])[0]

    def accuracy(self, X_test, y_test):
        return self.model.score(X_test, y_test)
'''
Создается конвейер с использованием make_pipeline, который включает в себя TfidfVectorizer для преобразования текста
 в векторы TF-IDF признаков и MultinomialNB для наивного байесовского классификатора. 
 При обучении модели, вызывается метод fit, который обучает модель на предоставленных обучающих данных.
'''

'''
TF-IDF (Term Frequency-Inverse Document Frequency) - это метод векторизации текстовых данных, который позволяет представить тексты в виде числовых векторов, 
путем присвоения каждому слову в тексте числового значения, основанного на его частоте в тексте и обратной частоте его встречаемости во всех текстах корпуса.

Частота терминов (Term Frequency, TF): Это отношение числа вхождений слова в документ к общему числу слов в документе. 
Это позволяет оценить важность слова в контексте конкретного документа.

Обратная частота документов (Inverse Document Frequency, IDF): Это логарифмическая обратная частота, 
с которой слово встречается в документах корпуса текстов. Слова, которые часто встречаются во всех документах, 
имеют низкую IDF, тогда как слова, которые встречаются редко, имеют высокую IDF. IDF уменьшает вес слова, 
которое встречается в большом количестве документов, и увеличивает вес слова, которое встречается в небольшом количестве документов.
'''

# Разделение на обучающий и тестовый наборы данных
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, train_size=0.8, random_state=42)

'''
Гиперпараметр alpha является параметром сглаживания в наивном байесовском классификаторе. 
Он добавляет "сглаживание Лапласа" к оценке вероятности каждого слова в документе для предотвращения нулевых вероятностей в случае,
 если слово отсутствует в обучающем наборе данных. Это помогает модели лучше обобщать на новые данные и снижает эффект переобучения.
'''

# Обучение модели на обучающем наборе данных
analyzer = SpamAnalyzer(alpha=0.1)  # Устанавливаем гиперпараметр alpha
analyzer.train(X_train, y_train)

# Оценка точности модели на тестовом наборе данных
accuracy = analyzer.accuracy(X_test, y_test) * 100
print("Model accuracy: {:.2f}%".format(accuracy))

# Добавляем два новых письма для проверки работы классификатора
new_emails = [
    "Arrival of goods! Burning discount! This week only! Come on in!",
    "Congratulations! You've won a free cruise. Claim your prize now!",
    "I hope you're doing well. Let's catch up soon."
]

# Классификация новых писем и вывод результатов
for email in new_emails:
    classification = "SPAM" if analyzer.classify(email) == 1 else "NOT SPAM"
    print("Email: '{}' is classified as '{}'.".format(email, classification))
