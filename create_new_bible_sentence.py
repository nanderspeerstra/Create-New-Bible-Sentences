# Small LSTM Network to Generate some Bible
import sys
import re

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
K.set_image_dim_ordering('tf') # or 'th' for theano

import argparse

########################################################################
#
########################################################################
# Input parser
parser = argparse.ArgumentParser(description='Create bible sentence')
parser.add_argument('--num_sentences', type=int, required=False,
                                        help='The number of sentences you want to generate.')
parser.add_argument('--len_sentences', type=int, required=False,
                                        help='The length of the sentences you want to generate.')
parser.add_argument('--models_location', type=str, required=True,
                                        help='The location of the models you created.')
args = parser.parse_args()


# Get model parameters
model_par = open(args.models_location+'/model_parameters.txt', 'r').readlines()
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
X = [thing.strip() for thing in open(args.models_location+'/dataX.txt', 'r').readlines()]
dataX = []
for thing in X:
    dataX.append([int(element.strip()) for element in thing.split(' ')])

# Read dataY
Y = open(args.models_location+'/dataY.txt', 'r').readlines()[0]
dataY = [int(thing) for thing in Y.split(' ')]

# one hot encode the output variable
y = np_utils.to_categorical(dataY)

# define the LSTM model
model = Sequential()
model.add(LSTM(parameters['n_hunits'], input_shape=(parameters['X_shape'][0], parameters['X_shape'][1])))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')

# load the network weights
filename = parameters['model_name']
model.load_weights(filename)
model.compile(loss='categorical_crossentropy', optimizer='adam')

# Load chars_to_int
int_to_char = {}
with open(args.models_location+'/chars_to_int.txt', 'r') as get_chars:
    
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


for sent in range(0,args.num_sentences):
        # pick a random seed
        start = numpy.random.randint(0, len(dataX)-1)
        pattern = dataX[start]
        print ("Seed:\t{}".format(''.join([int_to_char[value] for value in pattern])))
        
        output = ''
        # generate characters
        for i in range(args.len_sentences):
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
        
        output = re.sub('\n', ' ', output)    
        print(' - {}'.format(output))
        
print ("\n\nDone.")
