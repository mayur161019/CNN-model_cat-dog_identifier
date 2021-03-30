#DEPENDENCIES

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import cv2


#NETWORK_BLUIDING   #EPOCHS COUNT USED = 70

network = Sequential()
network.add(Conv2D(32, (3,3), input_shape = (64,64,3), activation='relu'))
network.add(MaxPooling2D(pool_size=(2,2)))

network.add(Conv2D(32, (3,3), activation='relu'))
network.add(MaxPooling2D(pool_size=(2,2)))

network.add(Flatten())

network.add(Dense(units = 3137, activation='relu'))
network.add(Dense(units = 3137, activation='relu'))
network.add(Dense(units = 2, activation='softmax'))   


network.compile(optimizer='Adam', loss='categorical_crossentropy', metrics = ['accuracy'])

history = network.fit_generator(train_dataset, epochs=epochs)

#TRAIN_TEST DATASET GENERATOR

training_generator = ImageDataGenerator(rescale=1./255,
                                        rotation_range=7,
                                        horizontal_flip=True,
                                        zoom_range=0.2)
train_dataset = training_generator.flow_from_directory('/content/cat_dog_2/training_set',
                                                        target_size = (64, 64),
                                                        batch_size = 32,
                                                        class_mode = 'categorical',
                                                        shuffle = True)

test_generator = ImageDataGenerator(rescale=1./255)
test_dataset = test_generator.flow_from_directory('/content/cat_dog_2/test_set',
                                                     target_size = (64, 64),
                                                     batch_size = 1,
                                                     class_mode = 'categorical',
                                                     shuffle = False)


#EVALUATION OF MODEL

predictions = network.predict(test_dataset)

predictions = np.argmax(predictions, axis = 1)

from sklearn.metrics import accuracy_score
accuracy_score(test_dataset.classes, predictions)

from sklearn.metrics import classification_report
print(classification_report(test_dataset.classes, predictions))