# Small LSTM Network to Generate some Bible
import sys

import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils

import tensorflow as tf
tf.python.control_flow_ops = tf  # some hack to get tf running with Dropout

from keras import backend as K
K.set_image_dim_ordering('th') # or 'th' for theano

# Get model parameters
model_par = open('model_parameters.txt', 'r').readlines()
parameters = {}
for line in model_par:
    line = line.strip()
    elements = line.split('\t')
    if len(elements)==3:
        parameters[elements[0]] = (int(elements[1]),int(elements[2]))
    elif elements[0] == 'model_name':
        parameters[elements[0]] = elements[1]
    else:
        parameters[elements[0]] = int(elements[1])

# Read dataX
X = [thing.strip() for thing in open('dataX.txt', 'r').readlines()]
dataX = []
for thing in X:
    dataX.append([int(element.strip()) for element in thing.split(' ')])

# Read dataY
Y = open('dataY.txt', 'r').readlines()[0]
dataY = [int(thing) for thing in Y.split(' ')]

# one hot encode the output variable
y = np_utils.to_categorical(dataY)

# define the LSTM model
model = Sequential()
model.add(LSTM(256, input_shape=(parameters['X_shape'][0], parameters['X_shape'][1]), return_sequences=True))
model.add(LSTM(256))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')

# load the network weights
filename = parameters['model_name']
model.load_weights(filename)
model.compile(loss='categorical_crossentropy', optimizer='adam')

# Load chars_to_int
int_to_char = {}
with open('chars_to_int.txt', 'r') as get_chars:
    
    for line in get_chars:
        try:
            line = line.strip('\n').split('\t')
            int_to_char[int(line[1])] = line[0]
        except IndexError:
            print('error')

# pick a random seed
start = numpy.random.randint(0, len(dataX)-1)
pattern = dataX[start]
print ("Seed:\n")
print (''.join([int_to_char[value] for value in pattern]))

output = ''
# generate characters
for i in range(100):
    x = numpy.reshape(pattern, (1, len(pattern), 1))
    x = x / float(parameters['n_vocab'])
    prediction = model.predict(x, verbose=0)
    index = numpy.argmax(prediction)
    result = int_to_char[index]
    seq_in = [int_to_char[value] for value in pattern]
    
    #sys.stdout.write(result)
    output += result
    
    pattern.append(index)
    pattern = pattern[1:len(pattern)]
    
print(output.strip())
print ("\n\nDone.")
