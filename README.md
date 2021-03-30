# CNN-model_cat-dog_identifier
convolutional neural network for identifying cats and dogs from images.


##Architecture

1. Convolutiona 2D layer with a specific requiring parameters such as filter:32, kernel_size: (3,3) image size = (64,64), activation='relu'
2. MaxPooling to reduce number of features with pool size (2,2).
3. Flatten layer to flatten matrix to vector so that it can be used in a dense layer.
4. Dense layer/Hidden Layer or a fully connected layer with a neurons of 3137.
5. finally added an output layer with a unit of 2 neurons(number of classes of datasets) and a softmax activation function.

##Image Augmentation

Due to the size of our dataset and class imbalance, we will not get the right accuracy. there is a need to increase the size of our dataset to get optimal result.

Keras ImageDataGenerator class is used to perform this operation.


##Hyperparameter

batch_size = 32
epoch = 70
optimizer = adam
loss = categorical_crossentropy
metrics = ['accuracy']


##Install Dependencies

1. Clone the repository on your system
2. Install the necsessary packages such as
   Python2 or Python3
   Tensorflow
   Keras
   Numpy
