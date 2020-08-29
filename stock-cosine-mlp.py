from convertdate import persian
from convertdate import iso
from convertdate import islamic
from convertdate import gregorian
from convertdate import utils
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import numpy as np
import keras
from scipy import spatial
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import train_test_split, StratifiedKFold
import tensorflow as tf
import os.path
from sklearn.metrics import classification_report, confusion_matrix
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
import yfinance as yf
from keras.models import load_model
from keras.applications import InceptionV3
from keras.models      import Model
filename = 'MCD'


data = yf.download(filename, start="2014-01-01", end="2020-04-22",
                   group_by="ticker",interval="1d")
pd.DataFrame(data).to_csv('Data/' + filename + '.csv')
stock = pd.read_csv('Data/' + filename + '.csv')
print("Preprocessing data ....")
print(stock)
newdata = np.array(stock)
first = str(newdata[len(newdata)-1][0])
first = first.split('-')
day = first[2]
mont = first[1]
year = first[0]
matris = []
temp = []
counter = 0
days_interval = 253
for i in range(len(newdata) - 1, -1 , -1):

        tmp = str(newdata[i][0])
        tmp = tmp.split('-')
        x1 = (tmp[2])
        x2 = (tmp[1])

        if ((counter == days_interval and i!= len(newdata) - 1) ):
            counter = 0
            matris.append(temp)
            temp = []
            temp.append(newdata[i][1:7])
        elif (i == 0 ):
            temp.append(newdata[i][1:7])
            matris.append(temp)
        else:
            temp.append(newdata[i][1:7])
        counter = counter + 1
del matris[-1]
zero = []
next_week = 20
window_size = 20
pred_dates = 1
avg_days = 1

for i in range(len(matris)):
    print(len(matris[i]))

windows = []
y_train = []
X_train = []

for j in range(0, days_interval - window_size, pred_dates):
    temp = []
    for k in range(0, len(matris), 1):
        temp.append(matris[k][j:j + (window_size)])
    windows.append(temp)
windows = np.array(windows)
# max_profit = 0
# for i in range(0, len(windows)):
#     if (i % (365-window_size) >= next_week):
#         sum = windows[i][0][0][4]
#         pred = 0
#         for ii in range(avg_days):
#             pred = pred + windows[i - next_week][0][ii][4]
#         pred = pred / avg_days
#         if (pred / sum > max_profit):
#             max_profit = pred / sum
# print("Max profit : " + str(max_profit))
ones = 0
zeros = 0
for i in range(0, len(windows)):
    if (i % (days_interval - window_size) >= next_week):
        X_train.append(windows[i])
        sum = windows[i][0][0][3]
        print(sum)
        pred = 0
        # for ii in range(avg_days):
        #     pred = pred + windows[i - next_week][0][ii][4]
        # pred = pred / avg_days
        # if (pred > sum):
        #     y_train.append(1)
        # else:
        #     y_train.append(0)
        t = 0
        z = 0
        for ii in range(avg_days):
            if (windows[i - next_week][0][ii ][3] > sum):
                t = t + 1
            if (windows[i - next_week][0][ii ][3] < sum):
                z = z + 1
        if (t == avg_days):
            zeros = zeros + 1
            y_train.append(0)
        elif (z == avg_days):
            ones = ones +1
            y_train.append(1)
        # else:
        #     y_train.append(0)
        print(y_train[-1])
close_windows = []
for i in range(next_week,len(windows)):
    array = []
    for j in range(len(matris)):
        tmp = []
        for k in range(window_size):
            tmp.append(windows[i][j][k][3])
        array.append(tmp)
    close_windows.append(array)
similar_windows = []
print("Similarity finding ...")
for i in range(len(close_windows)):
    temp = []
    temp.append(list(np.array(windows[i + next_week][0][:]).reshape(-1)))
    for k in range(1,len(matris)):
        closes_arr = []
        distance = []
        first = []
        second = []
        dist = 0
        jind = 0
        kind = 0
        for j in range(len(close_windows)):
            pattern = []
            pattern_first = []
            temp_window = close_windows[j][k][:]
            first = close_windows[i][0][:]
            for u in range(len(temp_window) - 1, 0 , -1):
                # pattern.append(temp_window[u-1] / temp_window[u])
                # pattern_first.append(first[u - 1] / first[u])
                if (first[u] > first[u-1]):
                    pattern_first.append(-1)
                else:
                    pattern_first.append(1)
                if (temp_window[u] > temp_window[u-1]):
                    pattern.append(-1)
                else:
                    pattern.append(1)
            dst = 1 - spatial.distance.cosine(pattern_first, pattern)
            # dst = 1 - spatial.distance.cosine(temp_window, first)
            if (dst > dist):
                dist = dst
                jind = j
                kind = k

        temp.append(list(np.array(windows[jind][kind][:]).reshape(-1)) )
        # Z = [x for _, x in sorted(zip(distance, closes_arr))]
        # temp.append(Z[len(Z)-1])
    similar_windows.append(temp)
    print(len(temp))
print("Done")

new_window = []
temp = []
temp.append(list(np.array(windows[0][0][:]).reshape(-1)))
for k in range(1,len(matris)):
    closes_arr = []
    distance = []
    first = []
    second = []
    dist = 0
    jind = 0
    kind = 0
    for j in range(len(close_windows)):
        pattern = []
        pattern_first = []
        temp_window = close_windows[j][k][:]
        first = close_windows[0][0][:]
        for u in range(len(temp_window) - 1, 0, -1):
            # pattern.append(temp_window[u-1] / temp_window[u])
            # pattern_first.append(first[u - 1] / first[u])
            if (first[u] > first[u - 1]):
                pattern_first.append(-1)
            else:
                pattern_first.append(1)
            if (temp_window[u] > temp_window[u - 1]):
                pattern.append(-1)
            else:
                pattern.append(1)
        dst = 1 - spatial.distance.cosine(pattern_first, pattern)
        # dst = 1 - spatial.distance.cosine(temp_window, first)
        if (dst > dist):
                dist = dst
                jind = j
                kind = k

    temp.append(list(np.array(windows[jind][kind][:]).reshape(-1)) )
        # Z = [x for _, x in sorted(zip(distance, closes_arr))]
        # temp.append(Z[len(Z)-1])
new_window = (temp)


train_Y_one_hot = to_categorical(y_train)
similar_windows.append(new_window)
similar_windows = np.array(similar_windows).reshape(-1, len(matris)*window_size*6)
similar_windows = pd.DataFrame(similar_windows)
scaler = preprocessing.StandardScaler()
scaled_values = scaler.fit_transform(similar_windows.iloc[:,:])
similar_windows.iloc[:,:] = scaled_values
similar_windows = np.array(similar_windows).reshape(-1, len(matris)*window_size*6)
new_window = similar_windows[-1]
similar_windows = list(similar_windows)
del similar_windows[-1]
similar_windows = np.array(similar_windows).reshape(-1, len(matris)*window_size*6)
train_X,valid_X,train_label,valid_label = train_test_split(similar_windows, train_Y_one_hot, test_size=0.3, random_state=13,shuffle=True)

X_train2 = []
y_train2 = []
X_val2 = []
y_val2 = []
for i in range(len(similar_windows) - 1 , -1 , -90):
    if ( i > 90) :
        for k in range(60):
            X_train2.append(similar_windows[i - k])
            y_train2.append(train_Y_one_hot[i-k])
        for j in range(30):
            X_val2.append(similar_windows[i - j - 60])
            y_val2.append(train_Y_one_hot[i- j - 60])
    else:
        for s in range(i , -1 , -1):
            X_train2.append(similar_windows[s])
            y_train2.append(train_Y_one_hot[s])
        break
X_train2 = np.array(X_train2)
y_train2 = np.array(y_train2)
X_val2 = np.array(X_val2)
y_val2 = np.array(y_val2)
np.random.shuffle(X_train2)
np.random.shuffle(y_train2)
np.random.shuffle(X_val2)
np.random.shuffle(y_val2)

if (os.path.exists("model"+filename+".h5") == False):
    kfold = KFold(n_splits=5, shuffle=True)
    fold_no = 1
    acc_per_fold = []
    loss_per_fold = []
    for train, test in kfold.split(similar_windows, train_Y_one_hot):
        epochs = 20
        model = Sequential()
        model.add(Dense(128, activation='relu', input_dim=(len(matris)*window_size*6)))
        model.add(Dropout(0.5))
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(2, activation='softmax'))
        model.compile(keras.optimizers.Adam(lr=.0001), loss='binary_crossentropy', metrics=['accuracy'])
        # model.fit(train_X, train_label,batch_size=64,epochs=epochs,verbose=1,validation_data=(valid_X, valid_label))
        model.fit(similar_windows[train], train_Y_one_hot[train], batch_size=64, epochs=epochs, verbose=1, validation_split=0.2)
        model.save("model"+filename+".h5")

        scores = model.evaluate(similar_windows[test], train_Y_one_hot[test], verbose=0)
        print(
            f'Score for fold {fold_no}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1] * 100}%')
        acc_per_fold.append(scores[1] * 100)
        loss_per_fold.append(scores[0])


        fold_no = fold_no + 1
        print('------------------------------------------------------------------------')
        print('Score per fold')
        for i in range(0, len(acc_per_fold)):
            print('------------------------------------------------------------------------')
            print(f'> Fold {i + 1} - Loss: {loss_per_fold[i]} - Accuracy: {acc_per_fold[i]}%')
        print('------------------------------------------------------------------------')
        print('Average scores for all folds:')
        print(f'> Accuracy: {np.mean(acc_per_fold)} (+- {np.std(acc_per_fold)})')
        print(f'> Loss: {np.mean(loss_per_fold)}')
        print('------------------------------------------------------------------------')

else:
    model = load_model("model"+filename+".h5")
    print(model.predict(new_window.reshape(-1, len(matris) * window_size * 6)))

print(ones)
print(zeros)
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
#
# df = pd.DataFrame (X_train)
# df2 = pd.DataFrame(train_Y_one_hot)
# filepath = 'btc.csv'
# filepath2 = 'btc_label.csv'
# df.to_csv(filepath, index=False)
# df2.to_csv(filepath2, index=False)
# print(len(matris))
