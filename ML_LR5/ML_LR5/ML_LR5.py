import pandas as pd
from io import StringIO
from apyori import apriori as apyori_apriori
import requests
from tabulate import tabulate

# Загружаем данные из URL
url = "https://raw.githubusercontent.com/adivyas99/Market-Basket-Optimization/master/Market_Basket.csv"
response = requests.get(url)

if response.status_code != 200:
    print("Не удалось загрузить данные. Код статуса:", response.status_code)
    exit(-1)
    
# Читаем данные с использованием pandas
data = pd.read_csv(StringIO(response.text), header=None)

# Заполняем пропуски с использованием ffill()
data.ffill(axis=1, inplace=True)

transactions = []
for i in range(0, len(data)):
    transactions.append([str(data.values[i, j]) for j in range(0, len(data.columns))])

# Используем библиотеку apyori для анализа Apriori
result = list(apyori_apriori(transactions, min_support=0.003, min_confidence=0.2, min_lift=4, min_length=2))

# Кастомная функция для сериализации
def serialize_record(record):
    return {
    'Items': list(record.items),
    'Support': record.support,
    'OrderedStatistics': [{
    'ItemsBase': list(ordered_stat.items_base),
    'ItemsAdd': list(ordered_stat.items_add),
    'Confidence': ordered_stat.confidence,
    'Lift': ordered_stat.lift
    } for ordered_stat in record.ordered_statistics]
    }

# Преобразование столбца 'OrderedStatistics' с помощью пользовательской функции
data_df = pd.DataFrame([serialize_record(record) for record in result])

# Отобразим результат в терминале с использованием tabulate
for _, row in data_df.iterrows():
    print(tabulate(row['OrderedStatistics'], headers='keys', tablefmt='pretty'))
    print()
