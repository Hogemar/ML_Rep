import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist  # библиотека базы выборок Mnist
from tensorflow import keras
from tensorflow.keras.layers import Dense, Flatten


(x_train, y_train), (x_test, y_test) = mnist.load_data()

# стандартизация входных данных
x_train = x_train / 255
x_test = x_test / 255

yes="Yes"
no="No"

y_train_last_digit_is_seven = np.array([yes if str(label)[-1] == '7' else no for label in y_train])
y_test_last_digit_is_seven = np.array([yes if str(label)[-1] == '7' else no for label in y_test])

model = keras.Sequential([

    Flatten(input_shape=(28, 28, 1)), # Выравнивание входных данных

    Dense(128, activation='relu'),   # Скрытый слой с 128 нейронами и функцией активации ReLU
    Dense(1, activation='sigmoid')  
])

model.compile(optimizer='adam',

              loss='binary_crossentropy',  # Функция потерь для бинарной  классификации
              metrics=['accuracy'])

model.fit(x_train, (y_train_last_digit_is_seven == yes).astype(int), batch_size=32, 
epochs=10, validation_split=0.2)

count_numb = 20
for n in range(count_numb):
    x = np.expand_dims(x_test[n], axis=0)
    res = model.predict(x)
    print(res)
    predicted_label = yes if res >= 0.5 else no
    print('Is the last digit 7? ' + str(predicted_label))
    plt.imshow(x_test[n], cmap=plt.cm.binary)
    plt.show()
    print()
