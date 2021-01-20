#!/usr/bin/env python
# coding: utf-8

# In[8]:


""" use kernel tensorflow_p36 - NOT TF 2"""
get_ipython().system('pip install --upgrade tensorflow==1.8.0')


# In[7]:


import cv2
import numpy as np
import os
import sys
import tensorflow as tf

from sklearn.model_selection import train_test_split

EPOCHS = 20
IMG_WIDTH = 224
IMG_HEIGHT = 224
NUM_CATEGORIES = 2
TEST_SIZE = 0.2


# In[13]:


def load_data(data_dir):
    """
    Load image data from directory `data_dir`.

    Assume `data_dir` has one directory named after each category, numbered
    0 through NUM_CATEGORIES - 1. Inside each category directory will be some
    number of image files.

    Return tuple `(images, labels)`. `images` should be a list of all
    of the images in the data directory, where each image is formatted as a
    numpy ndarray with dimensions IMG_WIDTH x IMG_HEIGHT x 3. `labels` should
    be a list of integer labels, representing the categories for each of the
    corresponding `images`.
    """
    images = []
    labels=[]
    counter = 0
    for labelname in os.listdir(data_dir):
        path= data_dir + "/" + labelname 
        for pic in os.listdir(path):
            path2= data_dir + "/" + labelname + "/" + pic
            img= cv2.imread(path2)
            #print ("path- ", path)
            #print ("path2- ", path2)
            re_img=cv2.resize(img,(IMG_WIDTH,IMG_HEIGHT))
            images.append(re_img)
            labels.append(labelname)
            counter= counter+1
    print(counter) 
    return (images,labels)


def get_model():
    """
    Returns a compiled convolutional neural network model. Assume that the
    `input_shape` of the first layer is `(IMG_WIDTH, IMG_HEIGHT, 3)`.
    The output layer should have `NUM_CATEGORIES` units, one for each category.
        """
# Create a convolutional neural network
    model = tf.keras.models.Sequential([

         # Convolutional layer. Learn 32 filters using a 3x3 kernel
         tf.keras.layers.Conv2D(
             32, (3, 3), activation="relu", padding = "same", input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)
         ),


         # Max-pooling layer, using 3x3 pool size
         tf.keras.layers.MaxPooling2D(pool_size=(4, 4)),
         tf.keras.layers.Dropout(0.2),
        
        tf.keras.layers.Conv2D(
             64, (3, 3), activation="relu", padding = "same", input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)
         ),


         # Max-pooling layer, using 3x3 pool size
         tf.keras.layers.MaxPooling2D(pool_size=(3, 3)),
         tf.keras.layers.Dropout(0.2),
         

         # Flatten units
         tf.keras.layers.Flatten(),
         tf.keras.layers.Dropout(0.2),
        
        

         # Add a hidden layer with dropout
         tf.keras.layers.Dense(NUM_CATEGORIES*32, activation="relu"),
         tf.keras.layers.Dropout(0.1),
        
         # Add a hidden layer with dropout
         tf.keras.layers.Dense(NUM_CATEGORIES*32, activation="relu"),
         tf.keras.layers.Dropout(0.1),
         
         # Flatten units
         tf.keras.layers.Flatten(),
         tf.keras.layers.Dropout(0.1),
        
        # Add a hidden layer with dropout
         tf.keras.layers.Dense(NUM_CATEGORIES*16, activation="relu"),
         tf.keras.layers.Dropout(0.1),
        
        # Add a hidden layer with dropout
         tf.keras.layers.Dense(NUM_CATEGORIES*16, activation="relu"),
         tf.keras.layers.Dropout(0.1),
        
        # Add a hidden layer with dropout
         tf.keras.layers.Dense(NUM_CATEGORIES*16, activation="relu"),
        
        # Add a hidden layer with dropout
         tf.keras.layers.Dense(NUM_CATEGORIES*32, activation="relu"),
         tf.keras.layers.Dropout(0.1),
         
         # Flatten units
         tf.keras.layers.Flatten(),
         tf.keras.layers.Dropout(0.1),
        
        # Add a hidden layer with dropout
         tf.keras.layers.Dense(NUM_CATEGORIES*16, activation="relu"),
         tf.keras.layers.Dropout(0.1),
        
        # Add a hidden layer with dropout
         tf.keras.layers.Dense(NUM_CATEGORIES*16, activation="relu"),
         tf.keras.layers.Dropout(0.1),
        
        

        # Add a hidden layer with dropout
         tf.keras.layers.Dense(NUM_CATEGORIES*8, activation="relu"),
         tf.keras.layers.Dropout(0.1),
         
         # Add an output layer with number of categories
         tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax")
     ])

# Train neural network
    model.compile(
         optimizer="adam",
         loss="categorical_crossentropy",
         metrics=["accuracy"]
   )

    return model


# In[14]:


# Get image arrays and labels for all image files
    images, labels = load_data("/home/ec2-user/SageMaker/BrainScans/brain")

    labels = tf.keras.utils.to_categorical(labels)
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(images), np.array(labels), test_size=TEST_SIZE)


# In[ ]:





# In[15]:


model = get_model()


# In[16]:


import time

start = time.time()

# Fit model on training data
model_history = model.fit(x_train, y_train, epochs=EPOCHS)

end = time.time()
print(end - start)


# In[17]:


# Evaluate neural network performance
 model.evaluate(x_test,  y_test, verbose=2)


# In[7]:


model.summary()


# In[8]:


model.save("model-latest1")

