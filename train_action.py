#!/usr/bin/env python
# coding: utf-8

# In[1]:


from __future__ import print_function
from keras.preprocessing import sequence
from keras.datasets import imdb
from attention.attention_keras import *
import pandas as pd
import numpy as np
from keras.models import Model
from keras.layers import *


# In[253]:


def abs_data(df, mind, lb):
    x = [abs(mind)+i for i in df[lb]]
    return x


# In[254]:


def cal_da(df, lb):
    x = df[lb].to_list()
    if min(x) < 0:
        return abs_data(df, min(x), lb)
    else:
        return x


# In[255]:


def get_data(df):
    ''''''
    lo = cal_da(df, 'longitude')
    la = cal_da(df, 'latitude')
    nc = cal_da(df, 'num_cloud')
    wd = cal_da(df, 'wind_direction')
    ws = cal_da(df, 'wind_speed')
    dp = cal_da(df, 'dew_point')
    t1 = cal_da(df, 't')
    s1 = cal_da(df, 'spi')
    
    rain = df['rain_six_hour'].to_list()
    
    x = list(zip(lo,la,nc,wd,ws,dp,t1 ,s1))
    y = np.array(rain)
    return x, y


# In[5]:


df1 = pd.read_csv('spi_train.csv')
df2 = pd.read_csv('spi_test.csv')


# In[6]:


ndf3 = df1.loc[df1['id'] != 58457]


# In[7]:


ndf4 = df2.loc[df2['id'] != 58457]


# In[8]:


print (len(ndf3), len(ndf4))


# In[9]:


ndf1 = df1[df1['id'] == 58457]
ndf2 = df2[df2['id'] == 58457]


# In[136]:


# new_df = pd.concat([ndf1, ndf3[:50003]],axis=0)
new_df = pd.concat([ndf1, ndf3[:1771]],axis=0)


# In[137]:


new_df2 = pd.concat([ndf2, ndf4[:3]],axis=0)
# new_df2 = ndf2


# In[138]:


print (len(new_df), len(new_df2))


# In[141]:


x_train, y_train = get_data(new_df)
x_test, y_test = get_data(new_df2)
# x_train, y_train = get_data(ndf1)
# x_test, y_test = get_data(ndf2)


# In[142]:


print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences',len(y_test))


# In[143]:


print('Pad sequences (samples x time)')
x_train = np.array(x_train)
x_test = np.array(x_test)
# x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
# x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)
print (x_train[0])


# In[144]:


from sklearn import preprocessing
import numpy as np

# n_y_train = y_train.reshape(13308,4)
n_y_train = y_train.reshape(1250,4)
n_y_test = y_test.reshape(246,4)
min_max_scaler = preprocessing.MinMaxScaler()
x1 = min_max_scaler.fit_transform(n_y_train)
x2 = min_max_scaler.fit_transform(n_y_test)
y_train = x1.flatten()
y_test = x2.flatten()
# print(y_train, y_test)


# In[131]:


max_features = 20000
maxlen = 8
batch_size = 32

S_inputs = Input(shape=(None,), dtype='float32')
print (S_inputs)
embeddings = Embedding(max_features,128)(S_inputs)
print (embeddings)
# embeddings = SinCosPositionEmbedding(128)(embeddings) # 增加Position_Embedding能轻微提高准确率
O_seq = Attention(8,16)([embeddings,embeddings,embeddings])
print (O_seq)
O_seq = GlobalAveragePooling1D()(O_seq)
O_seq = Dropout(0.5)(O_seq)
# outputs = Dense(1, activation='tanh')(O_seq)
outputs = Dense(1, activation='sigmoid')(O_seq)

model = Model(inputs=S_inputs, outputs=outputs)
# try using different optimizers and different optimizer configs
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


# In[145]:


print('Train...')
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=5,
          validation_data=(x_test, y_test))


# In[146]:


y_pred = model.predict(x_test, batch_size=32, verbose=0)


# In[151]:


import matplotlib.pyplot as plt


# In[192]:


nanjing_spi = ndf2['spi'].to_list()
# print (len(nanjing_spi))
x1 = np.arange(len(nanjing_spi))
plt.scatter(x1,nanjing_spi, c='r')  # 绘制数据点
plt.title('The SPI of NanJing')
plt.ylabel('SPI')
plt.xlabel('Num')
plt.savefig('spi.png')
# plt.show()


# In[223]:


x = [i for i in range(len(y_pred))]
# print (x)
# n_y_train = y_train.reshape(1250,4)
n_y_test = y_test.reshape(246,4)
n_y_pred = y_pred.reshape(246,4)
y1 = min_max_scaler.inverse_transform(n_y_test)
y2 = min_max_scaler.inverse_transform(n_y_pred)
y_test = y1.flatten()
y_pred = y2.flatten()
# print('data is ',data)
# print('after Min Max ',mm_data)
# print('origin data is ',origin_data


# In[236]:


import matplotlib.pyplot as plt
import numpy as np
np.random.seed(4)

fig = plt.figure()
ax1 = fig.add_subplot(111)

plt.title('The rain information of NanJing')

plot1 = ax1.plot(x[320:380], y_test[320:380], linewidth=2,linestyle=':', color='r', label='real')
plt.ylabel('Real Rainfall/mm')
ax2 = ax1.twinx()  # this is the important function

plot2 = ax2.plot(x[320:380], y_pred[320:380], linewidth=2, color='b', label='prediction')
plt.ylabel('Predict Rainfall/mm')
plt.xlabel('Sample Number')
lines = plot1 + plot2

ax1.legend(lines, [l.get_label() for l in lines]) # only need one legend definition
plt.savefig('test_pred2.png')


# In[247]:


import matplotlib.pyplot as plt
import numpy as np
np.random.seed(4)

fig = plt.figure()
ax1 = fig.add_subplot(111)

plt.title('The rain information of NanJing')

plot1 = ax1.plot(x, y_test, linewidth=2,linestyle=':', color='r', label='real')
plt.ylabel('Real Rainfall/mm')
ax2 = ax1.twinx()  # this is the important function

plot2 = ax2.plot(x, y_pred, linewidth=2, color='b', label='prediction')
plt.ylabel('Predict Rainfall/mm')
plt.xlabel('Sample Number')
lines = plot1 + plot2

ax1.legend(lines, [l.get_label() for l in lines]) # only need one legend definition
plt.savefig('test_pred.png')


# In[252]:


import matplotlib.pyplot as plt
import numpy as np
np.random.seed(4)

fig = plt.figure()
ax1 = fig.add_subplot(111)

plt.title('The rain information of NanJing')

plot1 = ax1.plot(x[200:400], y_test[200:400], linewidth=2,linestyle=':', color='r', label='real')
plt.ylabel('Rainfall/mm')
plt.xlabel('Sample Number')

ax2 = ax1.twinx()  # this is the important function

plot2 = ax2.plot(x[200:400], y_pred[200:400], linewidth=2, color='b', label='prediction')
# plt.ylabel('Predict Rainfall/mm')

plt.yticks([])
lines = plot1 + plot2

ax1.legend(lines, [l.get_label() for l in lines]) # only need one legend definition
plt.savefig('test_pred3.png')


# In[238]:


import matplotlib.pyplot as plt
import numpy as np
np.random.seed(4)

fig = plt.figure()
ax1 = fig.add_subplot(111)

plt.title('The rain information of NanJing')

plot1 = ax1.plot(y_pred[200:400], y_test[200:400], linewidth=2,linestyle=':', color='r', label='real')
plt.ylabel('Real Rainfall/mm')
plt.xlabel('Predict Rainfall/mm')
# ax2 = ax1.twinx()  # this is the important function

# plot2 = ax2.plot(x[200:400], y_pred[200:400], linewidth=2, color='b', label='prediction')
# plt.ylabel('Predict Rainfall/mm')
# plt.xlabel('Sample Number')
# lines = plot1 + plot2

# ax1.legend(lines, [l.get_label() for l in lines]) # only need one legend definition
plt.savefig('test_pred3.png')

