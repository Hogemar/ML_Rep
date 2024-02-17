import os.path
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation
from array import array

data_train = np.loadtxt('C:\Iris_Train.txt', delimiter = ';', dtype=np.float64)
X = data_train[:,[0,1,2]]
Y = data_train[:,[3]]

model = Sequential()
model.add(Dense(units = 8, input_shape = (3,)))
model.add(Activation("sigmoid"))
model.add(Dense(units = 24))
model.add(Activation("sigmoid"))
model.add(Dense(units = 12))
model.add(Activation("sigmoid"))
model.add(Dense(1))
model.add(Activation("sigmoid"))

model.compile(optimizer = 'RMSprop', loss = 'binary_crossentropy', metrics = ['accuracy'])
model.fit(X,Y, epochs = 1000, batch_size = 80)

#model.fit(X,Y, nb_epoch = 500, batch_size = 100)
data_test = np.loadtxt('C:\Iris_Test.txt', delimiter = ';', dtype=np.float64)
res = model.predict(data_test)

for var in res:
    if(abs(var[0] - 0.17) <= 0.17): var[0] = 1
    elif (abs(var[0] - 0.5) < 0.17): var[0] = 2
    elif (abs(var[0] - 0.83) <= 0.17): var[0] = 3
    
print(res)
