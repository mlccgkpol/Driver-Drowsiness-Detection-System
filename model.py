import os

import keras.layers
from keras.preprocessing import image
import matplotlib.pyplot as plt 
import numpy as np
from keras.utils.np_utils import to_categorical
import random,shutil
from keras.models import Sequential
from keras.layers import Dropout,Conv2D,Flatten,Dense, MaxPooling2D
from keras.models import load_model
import pandas as pd
import matplotlib.pyplot as plt


def generator(dir, gen=image.ImageDataGenerator(rescale=1./255), shuffle=True,batch_size=1,target_size=(28,28),class_mode='categorical'):

    return gen.flow_from_directory(dir,batch_size=batch_size,shuffle=shuffle,color_mode='grayscale',class_mode=class_mode,target_size=target_size)

BS= 32
TS=(28,28)
train_batch= generator('dataset/train',shuffle=True, batch_size=BS,target_size=TS)
valid_batch= generator('dataset/valid',shuffle=True, batch_size=BS,target_size=TS)
test_batch=generator('dataset/test',shuffle=True, batch_size=BS,target_size=TS)
train_size= len(train_batch.classes)//BS
val_size = len(valid_batch.classes)//BS

# img,labels= next(train_batch)
# print(img.shape)

model = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28,28,1)),
    keras.layers.BatchNormalization(),
    MaxPooling2D(pool_size=(1,1)),
    Conv2D(32,(3,3),activation='relu'),
    keras.layers.BatchNormalization(),
    MaxPooling2D(pool_size=(1,1)),
    Conv2D(64, (3, 3), activation='relu'),
    keras.layers.BatchNormalization(),
    MaxPooling2D(pool_size=(1,1)),
    Dropout(0.25),
    Flatten(),
    Dense(128, activation='relu'),
    keras.layers.BatchNormalization(),
    Dropout(0.5),
    Dense(2, activation='sigmoid')
])

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

history=model.fit(train_batch, validation_data=valid_batch,epochs=35,steps_per_epoch=train_size ,validation_steps=val_size)
pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1) # set the vertical range to [0-1]
plt.show()
model.save('models/model.h5', overwrite=True)
print(model.evaluate(test_batch))