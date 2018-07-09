
# coding: utf-8

# # convolutional Neural networks

# ## 1. load CIFAR-10 database

# In[4]:


import keras
from keras.datasets import cifar10

(x_train, y_train), (x_test, y_test) = cifar10.load_data()


# In[6]:


## 2. visulaize the first 24 training images
import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
fig = plt.figure(figsize=(20,5))
for i in range(36):
    ax = fig.add_subplot(3,12,i+1, xticks=[], yticks=[])
    ax.imshow(np.squeeze(x_train[i]))


# ## 3. Resclae the images by dividing every pixel in every image by 255

# In[7]:


x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255


# ## 4. Break dataset into training, testing and validation sets

# In[8]:


from keras.utils import np_utils
# one=hot encode the labels
num_classes = len(np.unique(y_train))
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

#break training set into training and validation sets
(x_train, x_valid) = x_train[5000:], x_train[:5000]
(y_train, y_valid) = y_train[5000:], y_train[:5000]

print('x_train shape:', x_train.shape)





# ## 5. define the model architecture

# In[9]:


from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
#define the model
model = Sequential()
# we need to flatten the input matrix, since MLPs only take vectores as input
model.add(Flatten(input_shape = x_train.shape[1:]))
model.add(Dense(1000, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(num_classes, activation='softmax'))
model.summary()


# ## 6. compile the model

# In[10]:


model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])


# ## 7. Train the model

# In[ ]:


from keras.callbacks import ModelCheckpoint

checkpointer = ModelCheckpoint(filepath='MLP.weights.best.hdf5', verbose=1, save_best_only=True)

hist = model.fit(x_train, y_train, batch_size=32, epochs=20, validation_data=(x_valid, y_valid), callbacks=[checkpointer], verbose=2, shuffle=True)


# ## 8. Load the model with best classification accuracy on the validation test

# In[ ]:


model.load_weights('MLP.weights.best.hdf5')


# ## 9. Calculate classification accuracy on test set

# In[ ]:


score = model.evaluate(x_test, y_test, verbose=0)
print('\n', "Test accuracy:", score[1])

