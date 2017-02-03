# Small LSTM Network to Generate some Bible
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
parser.add_argument('--num_hunits', type=int, required=True,
                                        help='The number of sentences you want to generate.')
parser.add_argument('--len_sequence', type=int, required=True,
                                        help='The length of the sentences you want to generate.')
parser.add_argument('--models_location', type=str, required=True,
                                        help='The location of the models you created.')
parser.add_argument('--num_epochs', type=int, required=True,
                                        help='The number of epochs for training.')
parser.add_argument('--file_name', type=str, required=True,
                                        help='The file name (txt) for training.')
args = parser.parse_args()


# load ascii text and covert to lowercase
filename = args.models_location+'/'+args.file_name
raw_text = open(filename).read()

# Create a smaller part to train on (for testing purposes)
#raw_text = raw_text[:2000]

# All characters in lowercase
raw_text = raw_text.lower()

# create mapping of unique chars to integers
chars = sorted(list(set(raw_text)))
char_to_int = dict((c, i) for i, c in enumerate(chars))
with open(args.models_location+'/chars_to_int.txt', 'w') as outChars:
    for key in char_to_int:
        outChars.write('{}\t{}\n'.format(key, char_to_int[key]))


# summarize the loaded data
n_chars = len(raw_text)
n_vocab = len(chars)
print ("Total Characters: ", n_chars)
print ("Total Vocab: ", n_vocab)

# prepare the dataset of input to output pairs encoded as integers
seq_length = args.len_sequence
dataX = []
dataY = []
for i in range(0, n_chars - seq_length, 1):
    seq_in = raw_text[i:i + seq_length]
    seq_out = raw_text[i + seq_length]
        	
    dataX.append([char_to_int[char] for char in seq_in])
    dataY.append(char_to_int[seq_out])
n_patterns = len(dataX)
print ("Total Patterns: ", n_patterns)


# Export dataX and dataY
with open(args.models_location+'/dataX.txt', 'w') as outX:
    for thing in dataX:
        line = ' '.join([str(element) for element in thing])
        outX.write(line + '\n')
        
with open(args.models_location+'/dataY.txt', 'w') as outY:
	line = ' '.join([str(element) for element in dataY])
	outY.write(line + '\n')  

# reshape X to be [samples, time steps, features]
X = numpy.reshape(dataX, (n_patterns, seq_length, 1))

# normalize
X = X / float(n_vocab)

# one hot encode the output variable
y = np_utils.to_categorical(dataY)

# define the LSTM model
model = Sequential()
model.add(LSTM(args.num_hunits, input_shape=(X.shape[1], X.shape[2])))#, return_sequences=True))
#model.add(LSTM(256))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')

# define the checkpoint
#filepath="weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
filepath = args.models_location+'/weights-best.hdf5'
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

# fit the model
model.summary()
model.fit(X, y, nb_epoch=args.num_epochs, batch_size=64, callbacks=callbacks_list)

# Write the parameters to a new file
model_out = open(args.models_location+'/model_parameters.txt', 'w')
model_out.write('n_hunits\t{}\n'.format(args.num_hunits))
model_out.write('n_chars\t{}\n'.format(n_chars))
model_out.write('n_vocab\t{}\n'.format(n_vocab))
model_out.write('seq_length\t{}\n'.format(seq_length))
model_out.write('n_patterns\t{}\n'.format(n_patterns))
model_out.write('X_shape\t{}\t{}\n'.format(X.shape[1], X.shape[2]))
model_out.write('model_name\t{}\n'.format(filepath))
model_out.close()
