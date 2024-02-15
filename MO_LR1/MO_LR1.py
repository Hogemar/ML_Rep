# import numpy as np

# # Функция активации
# def nonlin(x, deriv=False):
#     if (deriv == True):
#         return x*(1-x)
#     else:
#         return 1 / (1 + np.exp(-x))

# # Тренировочные наборы: одна строка – один набор
# X = np.array([ [0, 0, 1],
#                [0, 1, 1],
#                [1, 0, 1],
#                [1, 1, 1]] )
# #X = np.array([0, 0, 1])
# y = np.array([[0, 0, 1, 1]]).T # Ответы к каждому тренировочному набору
# #y = np.array([1]).T



# # Матрица весов от 3 входов (строки) к 4 нейронам (столбцы)
# syn0 = 2*np.random.random((3,4)) - 1
# # Матрица весов от 4 нейронов (строки) к 1 выходу
# syn1 = 2*np.random.random((4,1)) - 1

# l0 = X

# for iter in range(10000):
#     # Прямое распространение
#     l1 = nonlin(np.dot(l0, syn0)) # np.dot – матричное умножение. Получаем значения нейронов после акт. функции
#     l2 = nonlin(np.dot(l1, syn1)) # Получаем значения на выходе. Одна строка – один ответ на тренировочный набор

#     #print(f"{l1}\n\n{l2}")

#     # Вычисление ошибки и изменения весов
#     l2_error = y - l2                       # абсолютные ошибки ответов сети в сравнении с тестовыми ответами
#     l2_delta = l2_error * nonlin(l2, True)  # значение ошибки умножается на производную функ. акт.
    
#     #print(f"{l2_error}\n\n{l2_delta}")

#     l1_error = l2_delta.dot(syn1.T)
#     l1_delta = l1_error * nonlin(l1, deriv=True)

#     # Обновление весов
#     syn1 += l1.T.dot(l2_delta)
#     syn0 += l0.T.dot(l1_delta)
# #

# #print(l2)

# l1 = nonlin(np.dot(np.array([1,0,1]), syn0))
# l2 = nonlin(np.dot(l1, syn1))

# print(l2)



# import numpy as np

# # Функция активации
# def nonlin(x, deriv=False):
#     if (deriv == True):
#         return x*(1-x)
#     else:
#         return 1 / (1 + np.exp(-x))

# # Тренировочные наборы: одна строка – один набор
# X = np.array([ [0.15, 0.23],
#                [0.14, 0.32],
#                [0.412, 0.57],
#                [0.511, 0.22],
#                [0.36, 0.49]] )

# y = np.array([[0.38, 0.46, 0.982, 0.731, 0.85]]).T # Ответы к каждому тренировочному набору

# syn0 = 2*np.random.random((2,4)) - 1
# syn1 = 2*np.random.random((4,1)) - 1

# l0 = X

# for iter in range(10000):
#     # Прямое распространение
#     l1 = nonlin(np.dot(l0, syn0)) # np.dot – матричное умножение. Получаем значения нейронов после акт. функции
#     l2 = nonlin(np.dot(l1, syn1)) # Получаем значения на выходе. Одна строка – один ответ на тренировочный набор

#     #print(f"{l1}\n\n{l2}")

#     # Вычисление ошибки и изменения весов
#     l2_error = y - l2                       # абсолютные ошибки ответов сети в сравнении с тестовыми ответами
#     l2_delta = l2_error * nonlin(l2, True)  # значение ошибки умножается на производную функ. акт.
    
#     #print(f"{l2_error}\n\n{l2_delta}")

#     l1_error = l2_delta.dot(syn1.T)
#     l1_delta = l1_error * nonlin(l1, deriv=True)

#     # Обновление весов
#     syn1 += l1.T.dot(l2_delta)
#     syn0 += l0.T.dot(l1_delta)
# #

# #print(l2)

# l1 = nonlin(np.dot(np.array([0.3,0.3]), syn0))
# l2 = nonlin(np.dot(l1, syn1))

# print(l2)




import numpy as np

def nonlin(x, deriv=False):
    if (deriv == True):
        return x * (1 - x)
    else:
        return 1 / (1 + np.exp(-x))

X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])

y = np.array([[0, 1, 1, 1]]).T

syn0 = 2*np.random.random((2,4)) -1
syn1 = 2*np.random.random((4,1)) - 1

for iter in range(10000):
    # Прямое распространение
    l0 = X
    l1 = nonlin(np.dot(l0, syn0))
    l2 = nonlin(np.dot(l1, syn1))
    # Вычисление ошибки и изменения весов
    l2_error = y - l2
    l2_delta = l2_error * nonlin(l2, True)
    l1_error = l2_delta.dot(syn1.T)
    l1_delta = l1_error * nonlin(l1, deriv=True)
    # Обновление весов
    syn1 += l1.T.dot(l2_delta)
    syn0 += l0.T.dot(l1_delta)

for i in range(4):
    print(f"input: [{X[i][0]}; {X[i][1]}]; output: {l2[i]}\n")
