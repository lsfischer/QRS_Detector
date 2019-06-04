# importing required libraries
import numpy as np
import tensorflow as tf
from tensorflow.python.keras import layers
import matplotlib.pyplot as plt
import pickle as pkl
import random

# auxiliary function
# extracts from a long np-array (2-rows) a (2-rows)-random segment with a fixed length (seqL*ninputs)
def selectFrom1ecg(ecgBdata,seqL, ninputs, file_indexes):
    """
    x: An array with vairous files, channels and examples
    seqL:  number of timesteps to be used in recurrent nn
    ninput : is number of inputs in each timestep
    file_indexes: A list of the file indexes for training or validation set
    """
    segmentL  = seqL * ninputs
    numChan = 3
    
    inputs = np.array([])
    targets = []
    
    for idx, val in enumerate(file_indexes):
        
        inpOutSegment = tf.random_crop(ecgBdata[idx],[numChan, segmentL])
        channelII = inpOutSegment[0,:]
        channelV1 = inpOutSegment[1,:]
        target = inpOutSegment[2,:]
        
        concatenated_channels = np.concatenate((channelII, channelV1))
        
        #inputs = np.concatenate((inputs, concatenated_channels))
        targets.append(target)
        
    return inputs,target


dataset_array = []

files_not_to_read = [4,17,35,44,57,72,74]
index_counter = 0
for i in range(1, 76):
    
    if i not in files_not_to_read:
        file_path = f"./processed_data/Training/I{i:02}"
        file_data = pkl.load(open(file_path, "rb"))        
        index_counter = index_counter + 1
        
        info = [file_data["channelII"], file_data["channelV1"], file_data["label"]]
        info = np.array(info)
        info = info.astype(np.float32)
        dataset_array.append(info)

ecgs_array = np.array(dataset_array)


# number of examples
N = ecgs_array.shape[2]

# Sequence length (number of timesteps)
seqL = 20

# Sampling frequency
fs = 360

# For each timestep we give ninputs
ninputs = int(0.2*fs)

# We randomly select 35 files for the training set and the rest go to the validation set
#training_file_indexes = random.sample(list(range(68)), 35)
#validation_file_indexes = [x for x in range(68) if x not in training_file_indexes]
training_file_indexes = [1,2]
validation_file_indexes = [3,4]

# training data
# Create efficient training sequencess
trainData =tf.data.Dataset.from_tensors(ecgs_array)
#trainData = trainData.map(lambda x:  selectFrom1ecg(x, seqL, ninputs, training_file_indexes))
#trainData = trainData.repeat()  # Repeat the input indefinitely.
#batchSize = 8
#trainData = trainData.batch(batchSize)

#valData = tf.data.Dataset.from_tensors(ecgs_array[validation_file_indexes, :, :])
#valData = valData.map(lambda x:  selectFrom1ecg(x, seqL, ninputs, validation_file_indexes))
#valData = valData.repeat()  # Repeat the input indefinitely.
#batchSize = 8
#valData = valData.batch(batchSize)