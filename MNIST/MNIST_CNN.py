#!/usr/bin/env python
# coding: utf-8

# In[40]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.utils import np_utils
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
np.random.seed(10)


# In[41]:


# load data
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
X_train4D = X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32')
X_test4D = X_test.reshape(X_test.shape[0], 28, 28, 1).astype('float32')


# In[42]:


# normalize
X_train_normal = X_train4D / 255
X_test_normal = X_test4D / 255


# In[43]:


Y_train_onehot = np_utils.to_categorical(Y_train)
Y_test_onehot = np_utils.to_categorical(Y_test)


# In[44]:


# build CNN model
model = Sequential()
# conv_1: 16 (28x28)
model.add(Conv2D(filters=16, kernel_size=(5, 5), padding='same',
                 input_shape=(28, 28, 1), activation='relu'))

# 16 (28x28) -> 16 (14x14)
model.add(MaxPooling2D(pool_size=(2, 2)))

# conv_2: 36 (14x14)
model.add(Conv2D(filters=36, kernel_size=(5, 5),
                 padding='same', activation='relu'))

# 36 (14x14) -> 36 (7x7)
model.add(MaxPooling2D(pool_size=(2, 2)))

# ramdom dropout 25% neurons to avoid overfitting
model.add(Dropout(0.25))

# build flatten layer
model.add(Flatten())

# build hidden layer
model.add(Dense(128, activation='relu'))

# ramdom dropout 50% neurons to avoid overfitting
model.add(Dropout(0.5))

# build output layer
model.add(Dense(10, activation='softmax'))

print(model.summary())


# In[46]:


# training
# loss model: cross entropy
# optimizer method: adam
# evaluate methon: accuracy
model.compile(loss='categorical_crossentropy',
              optimizer='adam', metrics=['accuracy'])

# training rate(epochs): 10
# verbose = 2 : show training process
train_history = model.fit(x=X_train_normal, y=Y_train_onehot,
                          validation_split=0.2, epochs=10, batch_size=300, verbose=2)


# In[47]:


# %matplotlib inline

def show_train_history(train_history, train, validation):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title('Train History')
    plt.ylabel(train)
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()


# In[48]:


show_train_history(train_history, 'accuracy', 'val_accuracy')


# In[49]:


show_train_history(train_history, 'loss', 'val_loss')


# In[58]:


score = model.evaluate(X_test_normal, Y_test_onehot)
print('accuracy', score[1])


# In[55]:


# prediction
prediction = model.predict_classes(X_test_normal)


# In[56]:


prediction


# In[57]:

pd.crosstab(Y_test, prediction, rownames=['label'], colnames=['predict'])


# In[ ]:
