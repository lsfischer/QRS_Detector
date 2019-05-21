from keras import Input
from tensorflow import keras
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization, Conv2D, MaxPooling2D
from tensorflow.keras.layers import Activation, Flatten, Dropout, Dense

INIT_LR = 0.01
NUM_EPOCHS = 25
BS = 32

# Loading the data
((train_x, train_y), (test_x, test_y)) = keras.datasets.mnist.load_data()

# In our case we only have one input channel that is the black and white channel
train_x = train_x.reshape((train_x.shape[0], 28, 28, 1))
test_x = test_x.reshape((test_x.shape[0], 28, 28, 1))

# Conver to a number between 0 and 1
train_x = train_x.astype("float32") / 255.0
test_x = test_x.astype("float32") / 255.0


# We one-hot encode the trainning and testing labels
train_y = keras.utils.to_categorical(train_y, 10)
test_y = keras.utils.to_categorical(test_y, 10)


inputs = Input(shape=(28, 28, 1), name="inputs")
