from tkinter import *
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model

window = Tk()
window.title('Построение модели линейной регрессии')
window.geometry('400x200')
frame = Frame(window, padx=15, pady=15)
frame.pack(expand=True)
lb1 = Label(frame, text='Введите количество точек:')
lb1.grid(row=1, column=1)
tf1 = Entry(frame)
tf1.grid(row=1, column=2)
lb2 = Label(frame, text='Введите разброс точек:')
lb2.grid(row=2, column=1)
tf2 = Entry(frame)
tf2.grid(row=2, column=2)
n_features = 1

# Функция построения модели
def calc():
    k_noise = int(tf2.get())
    k_samples = int(tf1.get())
    X, y = datasets.make_regression(n_samples = k_samples, n_features=1, noise=k_noise)
    # Обучающие и тестовые данные
    train_size = int(k_samples * 0.8)
    test_size = int(k_samples * 0.2)
    # Нарезаем входные данные на обучающие и тестовые
    X_train = X[:-train_size]
    X_test = X[-test_size:]
    # Указываем, сколько будет выходных обучающих и тестовых данных
    y_train = y[:-train_size]
    y_test = y[-test_size:]
    # Создали регрессионную модель
    regr = linear_model.LinearRegression()
    # Обучение модели
    regr.fit(X_train, y_train)
    # Получение значений y
    y_pred = regr.predict(X_test)
    # Вывод картинки
    plt.scatter(X_test, y_test, color='purple', s=10)
    plt.plot(X_test, y_pred, color='black', linewidth=2)
    plt.title('Модель линейной регрессии')
    plt.xticks(())
    plt.yticks(())
    plt.savefig('model.png')
    plt.show()

btn = Button(frame, text="Построить модель", command=calc)
btn.grid(row=5, column=2)
window.mainloop()

