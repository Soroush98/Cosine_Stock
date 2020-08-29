from convertdate import persian
from convertdate import iso
from convertdate import islamic
from convertdate import gregorian
from convertdate import utils
import matplotlib.pyplot as plt
import numpy as np
import keras
import tensorflow as tf
from scipy import spatial
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
from tensorflow.python.ops.nn_ops import avg_pool

filename = 'saipa'
stock = pd.read_csv('Data/' + filename + '.csv')
stock = stock.drop_duplicates(['PDate'],keep='last')
stock = stock.reset_index(drop=True)

dates = stock.iloc[:,1]
newdata = []
#conv =  persian.to_gregorian(int(d[0]),int(d[1]),int(d[2]))
print("Filling Zeroes")
stock_fill = []
for i in range(0 , len(dates) , 1):
    if (stock.iloc[i][4] == 0):
        newstock = []
        index = i
        index2 = i
        while (stock.iloc[index][4] == 0):
            index = index + 1
        while (stock.iloc[index2][4] == 0):
            index2 = index2 - 1
        newstock.append(stock.iloc[i][0])
        newstock.append(stock.iloc[i][1])
        newstock.append(stock.iloc[i][2])
        newstock.append((stock.iloc[index][3] + stock.iloc[index2][3])/2)
        newstock.append((stock.iloc[index][4] + stock.iloc[index2][4]) / 2)
        newstock.append((stock.iloc[index][5] + stock.iloc[index2][5]) / 2)
        newstock.append((stock.iloc[index][6] + stock.iloc[index2][6]) / 2)
        newstock.append((stock.iloc[index][7] + stock.iloc[index2][7]) / 2)
        newstock.append((stock.iloc[index][8] + stock.iloc[index2][8])/2)
        newstock.append((stock.iloc[index][9] + stock.iloc[index2][9]) / 2)
        newstock.append((stock.iloc[index][10] + stock.iloc[index2][10]) / 2)
        newstock.append((stock.iloc[index][11] + stock.iloc[index2][11]) / 2)

        stock_fill.append(newstock)
    else:
        stock_fill.append(stock.iloc[i])
# df = pd.DataFrame (np.array(stock_fill))
# df.to_csv('saipa2.csv', index=False)
stock = pd.DataFrame(np.array(stock_fill))

print("Preprocessing data ....")
for i in range(len(dates) - 1 , -1 , -1):
    cur_Date = str(dates[i])
    day = cur_Date[6:8]
    mont = cur_Date[4:6]
    year = cur_Date[0:4]
    if (i != 0) :
        month = persian.monthcalendar(int(year),int(mont))
        flag = 0
        for j in range(len(month)):
            for k in range(len(month[j])):
                if (int(day) + 1 == month[j][k]):
                    flag = 1
        if (flag == 0):
            cur_Date2 = str(dates[i-1])
            day2 = cur_Date2[6:8]
            mont2 = cur_Date2[4:6]
            year2 = cur_Date2[0:4]
            newdata.append(stock.iloc[i])
            if (int(day2) != 1):
                dp = 1
                while (True):
                    if (int (day2) == dp):
                        break
                    newstock = []
                    temp = []
                    temp.append(year2)
                    temp.append(mont2)
                    temp.append(str(dp))

                    newdate = ''
                    for t in range(0,len(temp)):
                        newdate = newdate + temp[t]
                    newstock.append(stock.iloc[i - 1][0])
                    newstock.append(newdate)
                    newstock.append(stock.iloc[i - 1][2])
                    for l in range(3, len(stock.iloc[i - 1])):
                        newstock.append((stock.iloc[i - 1][l] + stock.iloc[i][l]) / 2)
                    newdata.append(newstock)
                    dp = dp + 1


        elif (flag == 1 ):
            cur_Date1 = str(dates[i - 1])
            day2 = cur_Date1[6:8]
            mont2 = cur_Date1[4:6]
            year2 = cur_Date1[0:4]
            newdata.append(stock.iloc[i])
            if (int(day2) != int(day) + 1 ):
                dp = int(day)
                first = 0
                while (True):
                    if (int (day2) == dp ):
                        break
                    flag = 0
                    for j in range(len(month)):
                        for k in range(len(month[j])):
                            if (dp + 1  == month[j][k]):
                                flag = 1
                    if (flag == 0 and  first == 0):
                        first = 1
                        dp = 0
                    dp = dp + 1
                    if (int(day2) == dp):
                        break
                    newstock = []
                    temp = []
                    temp.append(year2)
                    temp.append(mont2)
                    temp.append(str(dp))
                    newdate = ''
                    for t in range(0, len(temp)):
                        newdate = newdate + temp[t]
                    newstock.append(stock.iloc[i - 1][0])
                    newstock.append(newdate)
                    newstock.append(stock.iloc[i - 1][2])
                    for l in range(3, len(stock.iloc[i - 1])):
                        newstock.append((stock.iloc[i - 1][l] + stock.iloc[i][l]) / 2)
                    newdata.append(newstock)

newdata.append(stock.iloc[0])

newdata = np.array(newdata)
df = pd.DataFrame (newdata)
filepath = 'saipa2.csv'
df.to_csv(filepath, index=False)
first = str(newdata[len(newdata)-1][1])
day = int(first[6:8])
mont = int(first[4:6])
year = first[0:4]
matris = []
temp = []

for i in range(len(newdata) - 1, -1 , -1):

        tmp = str(newdata[i][1])
        x1 = int(tmp[6:8])
        x2 = int(tmp[4:6])
        if ((x1 == day and x2 == mont and i!= len(newdata) - 1) ):
            matris.append(temp)
            temp = []
            temp.append(newdata[i][3:12])
        elif (i == 0 ):
            temp.append(newdata[i][3:12])
            matris.append(temp)
        else:
            temp.append(newdata[i][3:12])
del matris[-1]
zero = []
window_size = 30
next_week  = 7
pred_dates = 1
avg_days = 7

for i in range(len(matris)):
    print(len(matris[i]))



windows = []
y_train = []
for j in range(0,365-window_size,pred_dates):
        temp = []
        for k in range(0, len(matris),1):
            temp.append(matris[k][j:j+(window_size)])
        windows.append(temp)
windows = np.array(windows)
print(len(windows))
# max_profit = 0
# for i in range(next_week, len(windows)):
#         sum = windows[i][0][0][3]
#         pred = 0
#         for ii in range(avg_days):
#             pred = pred + windows[i - next_week][0][ii][3]
#         pred = pred / avg_days
#         if (pred / sum > max_profit):
#             max_profit = pred / sum
# print("Max profit : " + str(max_profit))
for i in range(next_week, len(windows)):
        sum = windows[i][0][0][3]
        print(sum )
        pred = 0
        # for ii in range(avg_days):
        #     pred = pred + windows[i - next_week][0][ii][3]
        #     print(windows[i - next_week][0][ii][3])
        # pred = pred / avg_days
        t = 0
        z = 0
        for ii in range(avg_days):
            if ( windows[i-next_week][0][ii][3] >= sum ):
                t= t+1
            if ( windows[i-next_week][0][ii][3] <= sum ):
                z= z+1
        if (t == avg_days):
            y_train.append(1)
        elif (z == avg_days):
            y_train.append(2)
        else:
            y_train.append(0)
        #print(pred)
        # if (pred > sum):
        #     y_train.append(1)
        # if (pred / sum >= 1 + ((max_profit - 1)*2 / 4) and pred / sum <= max_profit):
        #     y_train.append(3)
        # elif (pred / sum  >= 1 +  ((max_profit-1)/4)  and pred / sum < 1 + ((max_profit-1)*2/4)):
        #     y_train.append(2)
        # elif (pred  >= sum  and pred / sum < 1 + ((max_profit-1)/4)):
        #     y_train.append(1)
        # elif (pred <= sum ):
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
            # closes_arr.append()
            dst = 1 - spatial.distance.cosine(pattern_first, pattern)
            # dst = 1 - spatial.distance.cosine(temp_window, first)
            if (dst > dist):
                # print(dst)
                dist = dst
                jind = j
                kind = k

        temp.append(list(np.array(windows[jind][kind][:]).reshape(-1)) )
        # print(close_windows[jind][k][:])
        # print(pattern)
        # print(close_windows[i][0][:])
        # print(pattern_first)
        # Z = [x for _, x in sorted(zip(distance, closes_arr))]
        # temp.append(Z[len(Z)-1])
    similar_windows.append(temp)
    print(len(temp))
print("Done")

train_Y_one_hot = to_categorical(y_train)
similar_windows = np.array(similar_windows).reshape(-1, len(matris)*window_size*9)
similar_windows = pd.DataFrame(similar_windows)
scaler = preprocessing.StandardScaler()
scaled_values = scaler.fit_transform(similar_windows.iloc[:,:])
similar_windows.iloc[:,:] = scaled_values
similar_windows = np.array(similar_windows).reshape(-1, len(matris)*window_size*9)
# train_X,valid_X,train_label,valid_label = train_test_split(similar_windows, train_Y_one_hot, test_size=0.3, random_state=13,shuffle=True)
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
epochs = 200

model = Sequential()
model.add(Dense(128, activation='relu', input_dim=(len(matris)*window_size*9)))
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(3, activation='softmax'))
model.compile(keras.optimizers.Adam(lr=.0001), loss='categorical_crossentropy', metrics=['accuracy'])
print(len(windows))
model = model.fit(X_train2, y_train2,batch_size=64,epochs=epochs,verbose=1,validation_data=(X_val2, y_val2))
# model = model.fit(train_X, train_label,batch_size=64,epochs=epochs,verbose=1,validation_data=(valid_X, valid_label))
# print(model.predict(new_window.reshape(-1,years_chunk*window_size*9)))

#print(model.evaluate(test_x,test_label,verbose=1))
accuracy = model.history['acc']
val_accuracy = model.history['val_acc']
loss = model.history['loss']
val_loss = model.history['val_loss']
epochs = range(len(accuracy))
plt.plot(epochs, accuracy, 'bo', label='Training accuracy' )
plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()
# df = pd.DataFrame (X_train)
# df2 = pd.DataFrame(train_Y_one_hot)
# filepath = 'tejarat.csv'
# filepath2 = 'tejarat_label.csv'
# df.to_csv(filepath, index=False)
# df2.to_csv(filepath2, index=False)
# print(len(matris))
