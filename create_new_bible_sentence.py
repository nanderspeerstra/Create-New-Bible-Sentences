# load the network weights
filename = "weights-improvement-04-1.6881.hdf5"
model.load_weights(filename)
model.compile(loss='categorical_crossentropy', optimizer='adam')

int_to_char = dict((i, c) for i, c in enumerate(chars))

# Read dataX
dataX = [thing.strip() for thing in open('dataX.txt', 'r').readlines()]
#dataY = [thing.strip() for thing in open('dataY.txt', 'r').readlines()]

# pick a random seed
start = numpy.random.randint(0, len(dataX)-1)
pattern = dataX[start]
print ("Seed:\n")
print (''.join([int_to_char[value] for value in pattern]))

# generate characters
for i in range(50):
	x = numpy.reshape(pattern, (1, len(pattern), 1))
	x = x / float(n_vocab)
	prediction = model.predict(x, verbose=0)
	index = numpy.argmax(prediction)
	result = int_to_char[index]
	seq_in = [int_to_char[value] for value in pattern]
	sys.stdout.write(result)
	pattern.append(index)
	pattern = pattern[1:len(pattern)]
print ("\n\nDone.")
