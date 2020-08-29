from convertdate import persian
from convertdate import iso
from convertdate import islamic
from convertdate import gregorian
from convertdate import utils
import matplotlib.pyplot as plt
import numpy as np
import keras
import tensorflow as tf
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from keras.applications import MobileNet
from keras.layers import Dense,GlobalAveragePooling2D
from keras.models import Sequential,Input,Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.utils import to_categorical
from keras.layers.advanced_activations import LeakyReLU
from sklearn.model_selection import train_test_split
import pandas as pd
from keras.applications import InceptionV3
from keras.models      import Model
filename = 'BTC-USD'

stock = pd.read_csv('Data/' + filename + '.csv')

#conv =  persian.to_gregorian(int(d[0]),int(d[1]),int(d[2]))
print("Preprocessing data ....")
print(stock)
newdata = np.array(stock)
first = str(newdata[len(newdata)-1][0])
first = first.split('/')
day = first[0]
mont = first[1]
year = first[2]
matris = []
temp = []

for i in range(len(newdata) - 1, -1 , -1):

        tmp = str(newdata[i][0])
        tmp = tmp.split('/')
        x1 = (tmp[0])
        x2 = (tmp[1])
        if ((x1 == day and x2 == mont and i!= len(newdata) - 1) ):
            matris.append(temp)
            temp = []
            temp.append(newdata[i][1:7])
        elif (i == 0 ):
            temp.append(newdata[i][1:7])
            matris.append(temp)
        else:
            temp.append(newdata[i][1:7])
del matris[-1]
zero = []
next_week = 21
window_size = 180
years_chunk = 4
pred_dates = 1
avg_days = 7

for i in range(len(matris)):
    print(len(matris[i]))

windows = []
y_train = []
X_train = []

for i in range(len(matris) - years_chunk + 1):
    for j in range(0,365-window_size,pred_dates):
        temp = []
        for k in range(i, i+years_chunk,1):
            temp.append(matris[k][j:j+(window_size)])
        windows.append(temp)
windows = np.array(windows)
max_profit = 0
for i in range(0, len(windows)):
    if (i % (365-window_size) >= next_week):
        sum = windows[i][0][0][4]
        pred = 0
        for ii in range(avg_days):
            pred = pred + windows[i - next_week][0][ii][4]
        pred = pred / avg_days
        if (pred / sum > max_profit):
            max_profit = pred / sum
print("Max profit : " + str(max_profit))
for i in range(0, len(windows)):
    if (i % (365 - window_size) >= next_week):
        X_train.append(windows[i])
        sum = windows[i][0][0][4]
        print(sum)
        pred = 0
        for ii in range(avg_days):
            pred = pred + windows[i - next_week][0][ii][4]
        pred = pred / avg_days
        if (pred / sum >= 1 + ((max_profit - 1)*2 / 4) and pred / sum <= max_profit):
            y_train.append(3)
        elif (pred / sum  >= 1 +  ((max_profit-1)/4)  and pred / sum < 1 + ((max_profit-1)*2/4)):
            y_train.append(2)
        elif (pred  >= sum  and pred / sum < 1 + ((max_profit-1)/4)):
            y_train.append(1)
        elif (pred <= sum ):
            y_train.append(0)

train_Y_one_hot = to_categorical(y_train)
X_train = np.array(X_train).reshape(-1, years_chunk*window_size*6)
X_train = pd.DataFrame(X_train)
scaler = preprocessing.StandardScaler()
scaled_values = scaler.fit_transform(X_train.iloc[:,:])
X_train.iloc[:,:] = scaled_values
X_train = np.array(X_train).reshape(-1, years_chunk*window_size*6)

train_X,valid_X,train_label,valid_label = train_test_split(X_train, train_Y_one_hot, test_size=0.3, random_state=13)
train_X = np.array(train_X)
train_X = train_X.reshape(train_X.shape[0],1,years_chunk*window_size*6)
train_label = np.array(train_label)
valid_X = np.array(valid_X)
valid_X = valid_X.reshape(valid_X.shape[0],1,years_chunk*window_size*6)
valid_label = np.array(valid_label)

epochs = 200
model = Sequential()
model.add(keras.layers.LSTM(input_dim=years_chunk*window_size*6,output_dim=50,return_sequences=True))
model.add(Dropout(0.2))
model.add(keras.layers.LSTM(100,return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(4,activation='softmax'))
model.compile(keras.optimizers.Adam(lr=.0001), loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_X, train_label, batch_size=256,epochs=epochs,verbose=1,validation_data=(valid_X, valid_label))
print(model.predict(np.array(windows[0]).reshape(1,1,years_chunk*window_size*6)))
# accuracy = model.history['acc']
# val_accuracy = model.history['val_acc']
# loss = model.history['loss']
# val_loss = model.history['val_loss']
# epochs = range(len(accuracy))
# plt.plot(epochs, accuracy, 'bo', label='Training accuracy' )
# plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy')
# plt.title('Training and validation accuracy')
# plt.legend()
# plt.figure()
# plt.plot(epochs, loss, 'bo', label='Training loss')
# plt.plot(epochs, val_loss, 'b', label='Validation loss')
# plt.title('Training and validation loss')
# plt.legend()
# plt.show()