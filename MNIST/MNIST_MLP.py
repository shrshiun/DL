#!/usr/bin/env python
# coding: utf-8

# In[42]:


import pandas as pd
from keras.layers import Dense
from keras.models import Sequential
import matplotlib.pyplot as plt
from keras.layers import Dropout
from keras.datasets import mnist
import numpy as np
from keras.utils import np_utils
np.random.seed(10)


# In[43]:


(X_train_image, Y_train_label), (X_test_image, Y_test_label) = mnist.load_data()


# In[44]:


print('image :', X_train_image.shape)
print('label:', Y_train_label.shape)


# In[45]:


def plot_images_labels_prediction(image, label, prediction, idx, num=10):
    fig = plt.gcf()
    fig.set_size_inches(12, 14)
    if num > 25:
        num = 25
    for i in range(0, num):
        ax = plt.subplot(5, 5, 1+i)
        ax.imshow(image[idx], cmap='binary')
        title = "label =" + str(label[idx])
        if len(prediction) > 0:  # 如果有傳入預測結果
            title += ",predict=" + str(prediction[idx])
        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        idx += 1
    plt.show()


# In[46]:


plot_images_labels_prediction(X_train_image, Y_train_label, [], 0, 15)


# In[47]:


# 將28x28 轉為一維(784)
X_train = X_train_image.reshape(60000, 784).astype('float32')
X_test = X_test_image.reshape(10000, 784).astype('float32')


# In[48]:


# Normalize (0~255) -> (0~1)
X_train_normal = X_train / 255
X_test_normal = X_test / 255


# In[49]:


# make label one-hot encoding
Y_train_onehot = np_utils.to_categorical(Y_train_label)
Y_test_onehot = np_utils.to_categorical(Y_test_label)


# In[66]:


# build model
# basic hidden layer: 256
# model = Sequential()
# ## 'unit':hidden layer ; 'input_dim':input layer
# model.add(Dense(units = 256,input_dim = 784,kernel_initializer = 'normal',activation = 'relu'))
# ## 'unit':output layer ; 'input_dim':input layer = 256(autofill)
# model.add(Dense(units = 10,kernel_initializer = 'normal',activation = 'softmax'))
# model.compile(loss = 'categorical_crossentropy',optimizer = 'adam',metrics=['accuracy'])

# hidden layer 256->1000
model = Sequential()
# 'unit':hidden layer ; 'input_dim':input layer
model.add(Dense(units=1000, input_dim=784,
                kernel_initializer='normal', activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(units=1000, input_dim=784,
                kernel_initializer='normal', activation='relu'))
model.add(Dropout(0.5))
# 'unit':output layer ; 'input_dim':input layer = 256(autofill)
model.add(Dense(units=10, kernel_initializer='normal', activation='softmax'))
model.compile(loss='categorical_crossentropy',
              optimizer='adam', metrics=['accuracy'])
model.summary()


# In[67]:


# training
# epochs:traing cycle ; verbose = 2: show traing process
train_history = model.fit(x=X_train_normal, y=Y_train_onehot,
                          validation_split=0.2, epochs=10, batch_size=200, verbose=2)


# In[69]:


# %matplotlib inline

def show_train_history(train_history, train, validation):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title('Train History')
    plt.ylabel(train)
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()


# In[70]:


show_train_history(train_history, 'accuracy', 'val_accuracy')


# In[71]:


show_train_history(train_history, 'loss', 'val_loss')


# In[72]:


# evaluate model with train data
score = model.evaluate(X_test_normal, Y_test_onehot)
print()
print('accuracy=', score[1])


# In[74]:


# predict
prediction = model.predict_classes(X_test)


# In[75]:


# show result of prediction
plot_images_labels_prediction(X_test_image, Y_test_label, prediction, idx=350)


# In[77]:


# build confusion matrix
pd.crosstab(Y_test_label, prediction, rownames=['label'], colnames=['predict'])


# In[78]:


# find error prediction
df = pd.DataFrame({'label': Y_test_label, 'predict': prediction})
df[(df.label == 9) & (df.predict == 4)]


# In[64]:


plot_images_labels_prediction(X_test_image, Y_test_label, prediction, 1232, 1)


# In[ ]:
