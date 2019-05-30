import os
import numpy as np
from tensorflow import keras
from keras.optimizers import SGD
from keras.models import Model
from keras.layers import Input, BatchNormalization, Conv2D, MaxPooling2D
from keras.layers import Activation, Flatten, Dropout, Dense

# Number of epochs and batch size to user in the old model
INIT_LR = 0.01
NUM_EPOCHS = 5
BS = 512

# We load the mnist data instead of the fashion_mnist data
((train_x, train_y), (test_x, test_y)) = keras.datasets.mnist.load_data()

# In this case we only have one input channel that is the black and white channel
train_x = train_x.reshape((train_x.shape[0], 28, 28, 1))
test_x = test_x.reshape((test_x.shape[0], 28, 28, 1))

train_x = train_x.astype("float32") / 255.0
test_x = test_x.astype("float32") / 255.0

# We one-hot encode the trainning and testing labels
train_y = keras.utils.to_categorical(train_y, 10)
test_y = keras.utils.to_categorical(test_y, 10)

# Defining the model using Keras functional API

# Functional API is better for transfer learning as it supports the concatenation
# of neural networks in a Concatenation Layer object
# The sequential API is a more rigid API to defining neural networks

# We create an input object that works like tf.placeholder
# We add a name to the object so we can find it more easily
inputs = Input(shape=(28, 28, 1), name="inputs")

# The next part of the model definition is the same as in Lab3 exercise
# Note that these objects implement the __call__ method and can be interpreted as functions
# The input they receive is the input the layer receives
layer = Conv2D(32, (3, 3), padding="same", input_shape=(28, 28, 1))(inputs)
layer = Activation("relu")(layer)
layer = BatchNormalization(axis=-1)(layer)

# The input layer is a convolutional layer with 32 filters
# The shape of the kernel in this layer is 3x3
# We add padding in this layer (so we can start the kernel right at the beginning of the image)
# and in this case we use padding "same" for it to add values to the padding that are copied from the original matrix (it could also be 0)
layer = Conv2D(32, (3, 3), padding="same")(layer)

# For this layer we add a ReLU activation
# We need to add ReLU because a convolution is still a linear transformation
# so we add ReLU for it to be a non linear transformation
layer = Activation("relu")(layer)

# We add batch normalization here
# This normalizes the output from the previous layer in order
# for the input of the next layer to be normalized
# In this case we put the channels at the end so we don't need to specify the axis of normalization
# otherwise we would need to specify
layer = BatchNormalization(axis=-1)(layer)

# In this layer we Pool the layer before in order to reduce the number of features
# Since we are using a 2x2 pooling size we are keeping only half of the features in each dimension
# So instead of a 28*28 vector we now have a 14*14 tensor
# Since we are omitting the stride Keras assumes the same stride as pool size which is what we want
layer = MaxPooling2D(pool_size=(2, 2))(layer)

# We add a dropout layer of 25% dropout for regularization
layer = Dropout(0.25)(layer)

# We add another convolution layer, in this case we don't need to specify the input shape
# because keras finds out the right input shape
layer = Conv2D(64, (3, 3), padding="same")(layer)
layer = Activation("relu")(layer)
layer = BatchNormalization(axis=-1)(layer)

layer = Conv2D(64, (3, 3), padding="same")(layer)
layer = Activation("relu")(layer)
layer = BatchNormalization(axis=-1)(layer)

# After this pooling we have a 7*7 tensor
layer = MaxPooling2D(pool_size=(2, 2))(layer)
layer = Dropout(0.25)(layer)

# We add a Flatten layer in order to transform the input tensor into a vector
# In this case we had a 7*7*64 (7*7*the number of filters we have)
features = Flatten(name="features")(layer)

# Fully connected part of the network
layer = Dense(512)(features)
layer = Activation("relu")(layer)
layer = BatchNormalization()(layer)
layer = Dropout(0.5)(layer)
layer = Dense(10)(layer)
layer = Activation("softmax")(layer)

# Here we say where the model starts and ends
old_model = Model(inputs=inputs, outputs=layer)

old_model.compile(optimizer=SGD(lr=INIT_LR, momentum=0.9, decay=INIT_LR / NUM_EPOCHS), loss="categorical_crossentropy",
                  metrics=["accuracy"])

tensorboard_callback = keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0,
                                                   write_graph=True, write_images=True)

if len(os.listdir("./files")) == 0:
    # If there are no files in the Files directory
    # it means that the network hasn't been trained yet, so we
    # need to train it and save its weights

    history = old_model.fit(train_x, train_y, validation_data=(test_x, test_y), batch_size=BS, epochs=NUM_EPOCHS,
                            callbacks=[tensorboard_callback])

    # This saves the weights to the specified file in HDF5 format
    old_model.save_weights('./files/mnist_model.h5')

# Loading the weights previously obtained by training the network
old_model.load_weights("./files/mnist_model.h5")

# We now iterate over all the layers in the model
# In order to freeze them, we don't want to train this model
for layer in old_model.layers:
    layer.trainable = False

# Now we create the new model, that will take advantage of the old models structure

layer = Dense(512)(old_model.get_layer("features").output)
layer = Activation("relu")(layer)
layer = BatchNormalization()(layer)
layer = Dropout(0.5)(layer)
layer = Dense(256)(layer)
layer = Activation("relu")(layer)
layer = BatchNormalization()(layer)
layer = Dropout(0.2)(layer)
layer = Dense(26)(layer)
layer = Activation("softmax")(layer)

# Here we say that the model starts where the old model ends
# and ends in the layer object
model = Model(inputs=old_model.get_layer("inputs").output, outputs=layer)

model.compile(optimizer=SGD(lr=1e-2, momentum=0.9), loss="categorical_crossentropy",
              metrics=["accuracy"])

# Loading the new data
new_train_x = np.load('./data/imagesLettersTrain.npy')
new_train_y = np.load('./data/labelsTrain.npy')

new_test_x = np.load('./data/imagesLettersTest.npy')
new_test_y = np.load('./data/labelsTest.npy')

# In this case we only have one input channel that is the black and white channel
new_train_x = new_train_x.reshape((new_train_x.shape[0], 28, 28, 1))
new_test_x = new_test_x.reshape((new_test_x.shape[0], 28, 28, 1))

new_train_x = new_train_x.astype("float32") / 255.0
new_test_x = new_test_x.astype("float32") / 255.0

# We one-hot encode the trainning and testing labels
# Now we have 26 different labels so we one-hot encode a vector with size 26
new_train_y = keras.utils.to_categorical(new_train_y, 26)
new_test_y = keras.utils.to_categorical(new_test_y, 26)

NEW_NUM_EPOCHS = 10
NEW_BS = 20

fitting = model.fit(new_train_x, new_train_y, validation_data=(new_test_x, new_test_y), batch_size=NEW_BS,
                    epochs=NEW_NUM_EPOCHS, callbacks=[tensorboard_callback])

# TODO We also need to train a network from scratch to compared with these results
