# import numpy as np
# import pandas as pd
# from sklearn.decomposition import TruncatedSVD
# from sklearn.metrics import mean_squared_error
# from math import sqrt
#
# # �������� ������ �� ����� CSV
# file_path = 'ratings_7.csv'
# data = pd.read_csv(file_path, header=None)
#
# # ���������������� ������ ��� ���������� ������� ������������ ��������� � �������������
# ratings_matrix = data.values.T
# # # ���������� SVD ��� ��������� ��������� ������ ��������� � �������������
# svd = TruncatedSVD(n_components=9)  # �������� ����� ������� ��������
# items, D, users = svd.fit_transform(ratings_matrix), np.diag(svd.singular_values_), svd.components_
# #
# # # ���������� ������������� ������ ��� �������������
# predicted_ratings = np.dot(items, np.dot(D, users))
# #
# # # ������ �������� ���������������� ������� � ������� RMSE
# rmse = sqrt(mean_squared_error(ratings_matrix, predicted_ratings))
#
# print("RMSE for item-based recommendation system:", rmse) 8339.435426992264 - ����� ������


""""""

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from math import sqrt

# �������� ������ �� ����� CSV
file_path = 'ratings_7.csv'
data = pd.read_csv(file_path, header=None)

# ���������� ������ ��� ���������� ������� ������������ ��������� � �������������
ratings_matrix = data.values.T  # ������������� ������� ��� item-based �������

# ������������ ������
scaler = StandardScaler()
normalized_ratings_matrix = scaler.fit_transform(ratings_matrix)

# ����� ������������ � �������� ��� ������� ����� �������� ������
user_id = 0
item_id = 3

# �������� ��������� ���������� ���������
for n_components in range(1, 20, 2):
    svd = TruncatedSVD(n_components=n_components)
    users, D, items = svd.fit_transform(normalized_ratings_matrix), np.diag(svd.singular_values_), svd.components_
    predicted_ratings = np.dot(users, np.dot(D, items))
    # �������� ���������������
    predicted_ratings = scaler.inverse_transform(predicted_ratings)
    # ������������ RMSE �� �������� �����
    rmse = sqrt(mean_squared_error(ratings_matrix, predicted_ratings))
    print(f"RMSE with {n_components} components:", rmse)

    # ������� ������ ��� �������������� ������������ � ��������
    print(f"Expected rating for user {user_id} and item {item_id}:", predicted_ratings[user_id, item_id])
    print()


""""""

# ��������� ��������� ������ TruncatedSVD � ��������� ������ ������� ��������.
#
# ����� fit_transform() ����������� � ��������������� ������� normalized_ratings_matrix,
# ����� �������� ��������� ������� users � items, � ����� ������ ����������� �������� D.
#
# ������������� ������ �������������� ����� ��������� ��������� ������ users, D � items.
# �������������� RMSE (������������������ ������) ����� ��������������� �������� normalized_ratings_matrix �
# �������������� ��������.


    # RMSE with 1 components: 9.524381882302324
    # RMSE with 5 components: 20.85928198670049
    # RMSE with 9 components: 27.799810962092025
    # RMSE with 13 components: 33.13103973967289
    # RMSE with 17 components: 37.6674144670338


#��� � �������������� ������ K ��������� �������
# import numpy as np
# import pandas as pd
# from sklearn.neighbors import NearestNeighbors
# from sklearn.metrics import mean_squared_error
# from math import sqrt
# 
# # �������� ������ �� ����� CSV
# file_path = 'ratings_7.csv'
# data = pd.read_csv(file_path, header=None)
# 
# # ���������������� ������ ��� ���������� ������� ������������ ��������� � �������������
# ratings_matrix = data.values.T
# 
# # ������������� ������ K ��������� �������
# k_neighbors = 5
# knn_model = NearestNeighbors(n_neighbors=k_neighbors)
# 
# # �������� ������ �� ��������������� ������
# knn_model.fit(ratings_matrix)
# 
# # ����� ��������� ������� ��� ������� ������������
# distances, indices = knn_model.kneighbors(ratings_matrix)
# 
# # ������� �������� ������ ��������� ������� ��� ������������
# predicted_ratings_knn = np.zeros_like(ratings_matrix)
# for i in range(ratings_matrix.shape[0]):
#     nearest_neighbors_ratings = ratings_matrix[indices[i], :]
#     predicted_ratings_knn[i, :] = np.mean(nearest_neighbors_ratings, axis=0)
# 
# # ������ �������� ������ � ������� RMSE
# rmse_knn = sqrt(mean_squared_error(ratings_matrix, predicted_ratings_knn))
# print("RMSE with KNN:", rmse_knn)
# 
# # ����� ��������� ������ ��������
# item_id = 2 # ����� ��������, ��� �������� ����� ������� ��������� ������
# 
# nearest_neighbors_ratings_item = ratings_matrix[indices[:, item_id], item_id]
# expected_rating_item = np.mean(nearest_neighbors_ratings_item)
# print(f"Expected rating for item {item_id}: {expected_rating_item}") #RMSE with KNN: 0.8942751254507754
