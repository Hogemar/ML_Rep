# import numpy as np
# import pandas as pd
# from sklearn.decomposition import TruncatedSVD
# from sklearn.metrics import mean_squared_error
# from math import sqrt
#
# # Загрузка данных из файла CSV
# file_path = 'ratings_7.csv'
# data = pd.read_csv(file_path, header=None)
#
# # Транспонирование данных для построения матрицы предпочтений элементов к пользователям
# ratings_matrix = data.values.T
# # # Применение SVD для получения урезанных матриц элементов и пользователей
# svd = TruncatedSVD(n_components=9)  # Выбираем число скрытых факторов
# items, D, users = svd.fit_transform(ratings_matrix), np.diag(svd.singular_values_), svd.components_
# #
# # # Вычисление предсказанных оценок для пользователей
# predicted_ratings = np.dot(items, np.dot(D, users))
# #
# # # Оценка точности рекомендательной системы с помощью RMSE
# rmse = sqrt(mean_squared_error(ratings_matrix, predicted_ratings))
#
# print("RMSE for item-based recommendation system:", rmse) 8339.435426992264 - фигня полная


""""""

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from math import sqrt

# Загрузка данных из файла CSV
file_path = 'ratings_7.csv'
data = pd.read_csv(file_path, header=None)

# Подготовка данных для построения матрицы предпочтений элементов к пользователям
ratings_matrix = data.values.T  # Транспонируем матрицу для item-based подхода

# Нормализация данных
scaler = StandardScaler()
normalized_ratings_matrix = scaler.fit_transform(ratings_matrix)

# Номер пользователя и предмета для которых будем выводить оценки
user_id = 0
item_id = 3

# Прогоним различное количество компонент
for n_components in range(1, 20, 2):
    svd = TruncatedSVD(n_components=n_components)
    users, D, items = svd.fit_transform(normalized_ratings_matrix), np.diag(svd.singular_values_), svd.components_
    predicted_ratings = np.dot(users, np.dot(D, items))
    # Обратное масштабирование
    predicted_ratings = scaler.inverse_transform(predicted_ratings)
    # Рассчитываем RMSE на исходной шкале
    rmse = sqrt(mean_squared_error(ratings_matrix, predicted_ratings))
    print(f"RMSE with {n_components} components:", rmse)

    # Выводим оценку для фиксированного пользователя и предмета
    print(f"Expected rating for user {user_id} and item {item_id}:", predicted_ratings[user_id, item_id])
    print()


""""""

# Создается экземпляр класса TruncatedSVD с указанным числом скрытых факторов.
#
# Метод fit_transform() применяется к нормализованной матрице normalized_ratings_matrix,
# чтобы получить урезанные матрицы users и items, а также вектор сингулярных значений D.
#
# Предсказанные оценки рассчитываются путем умножения урезанных матриц users, D и items.
# Рассчитывается RMSE (среднеквадратичная ошибка) между нормализованной матрицей normalized_ratings_matrix и
# предсказанными оценками.


    # RMSE with 1 components: 9.524381882302324
    # RMSE with 5 components: 20.85928198670049
    # RMSE with 9 components: 27.799810962092025
    # RMSE with 13 components: 33.13103973967289
    # RMSE with 17 components: 37.6674144670338


#код с использованием модели K ближайших соседей
# import numpy as np
# import pandas as pd
# from sklearn.neighbors import NearestNeighbors
# from sklearn.metrics import mean_squared_error
# from math import sqrt
# 
# # Загрузка данных из файла CSV
# file_path = 'ratings_7.csv'
# data = pd.read_csv(file_path, header=None)
# 
# # Транспонирование данных для построения матрицы предпочтений элементов к пользователям
# ratings_matrix = data.values.T
# 
# # Инициализация модели K ближайших соседей
# k_neighbors = 5
# knn_model = NearestNeighbors(n_neighbors=k_neighbors)
# 
# # Обучение модели на нормализованных данных
# knn_model.fit(ratings_matrix)
# 
# # Поиск ближайших соседей для каждого пользователя
# distances, indices = knn_model.kneighbors(ratings_matrix)
# 
# # Среднее значение оценок ближайших соседей для предсказания
# predicted_ratings_knn = np.zeros_like(ratings_matrix)
# for i in range(ratings_matrix.shape[0]):
#     nearest_neighbors_ratings = ratings_matrix[indices[i], :]
#     predicted_ratings_knn[i, :] = np.mean(nearest_neighbors_ratings, axis=0)
# 
# # Оценка точности модели с помощью RMSE
# rmse_knn = sqrt(mean_squared_error(ratings_matrix, predicted_ratings_knn))
# print("RMSE with KNN:", rmse_knn)
# 
# # Вывод ожидаемой оценки предмета
# item_id = 2 # Номер предмета, для которого нужно вывести ожидаемую оценку
# 
# nearest_neighbors_ratings_item = ratings_matrix[indices[:, item_id], item_id]
# expected_rating_item = np.mean(nearest_neighbors_ratings_item)
# print(f"Expected rating for item {item_id}: {expected_rating_item}") #RMSE with KNN: 0.8942751254507754
