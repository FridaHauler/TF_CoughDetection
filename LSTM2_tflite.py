from numpy import mean
from numpy import std
from numpy import dstack
from pandas import read_csv
from keras.models import Sequential, save_model
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import TimeDistributed
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.utils import to_categorical
from matplotlib import pyplot
from tensorflow.keras import Model
import tensorflow as tf
from tensorflow import lite

import os, re, shutil

def getFileNamesFromDir(folder,prefix):
    fileRegex = re.compile(r'([a-zA-Z0-9_])%s\.csv' % prefix)
    folder = os.path.abspath(folder)
    # FileNotFoundError: [Errno 2] No such file or directory:   '*_<prefix>.csv'
    os.chdir(folder)  # This line is to avoid the FileNotFoundError.
    # Make a list to contain the file with prefix.
    fileNames = list()
    for filename in os.listdir(folder):
        if re.search(fileRegex, filename):
            fileNames.append(filename)
            print('fileName.....', filename)
    return fileNames



# load a single file as a numpy array
def load_file(filepath):
	dataframe = read_csv(filepath, header=None, delim_whitespace=True)
	return dataframe.values

# load a list of files and return as a 3d numpy array
def load_group(filenames, prefix=''):
	loaded = list()
	for name in filenames:
		data = load_file(prefix + name)
		loaded.append(data)
	# stack group so that features are the 3rd dimension
	loaded = dstack(loaded)
	#print('loaded groups stacked: ####### ', loaded)
	return loaded

# load a dataset group, such as train or test
def load_dataset_group(group, prefix=''):
	filepath = prefix + group + '/Inertial Signals/'
	# load all 9 files as a single array
	filenames = list()
	myfilenames = list()
	# total acceleration
	filenames += ['total_acc_x_'+group+'.txt', 'total_acc_y_'+group+'.txt', 'total_acc_z_'+group+'.txt']
	#print('filenames total: ', filenames)
	# body acceleration
	filenames += ['body_acc_x_'+group+'.txt', 'body_acc_y_'+group+'.txt', 'body_acc_z_'+group+'.txt']
	# body gyroscope
	filenames += ['body_gyro_x_'+group+'.txt', 'body_gyro_y_'+group+'.txt', 'body_gyro_z_'+group+'.txt']
	# load input data
	X = load_group(filenames, filepath)
	#print('allfilenames: ',filenames, 'path: ', filepath)
	# load class output
	y = load_file(prefix + group + '/y_'+group+'.txt')
	### stuff to mess around with:
	mypath = 'c:\\Users\\frida.hauler\\Anaconda3\\myTries\dataSets\\'
	myfilenames += [getFileNamesFromDir(mypath, "frames" )]
	xx=load_group(myfilenames, mypath)
	print('$$$$$$$$$$$$$$$', xx, type(myfilenames), 'vs filenames" ', X, type(filenames) )
	return X, y

# load the dataset, returns train and test X and y elements
def load_dataset(prefix=''):
	# load all train
	print('**************************************', prefix)
	trainX, trainy = load_dataset_group('train', prefix + 'HARDataset/')
	#print(trainX.shape, trainy.shape)
	# load all test
	testX, testy = load_dataset_group('test', prefix + 'HARDataset/')
	print(testX.shape, testy.shape)
	# zero-offset class values
	trainy = trainy - 1
	testy = testy - 1
	# one hot encode y
	trainy = to_categorical(trainy)
	testy = to_categorical(testy)
	print(trainX.shape, trainy.shape, testX.shape, testy.shape)
	return trainX, trainy, testX, testy

# fit and evaluate a LSTM ReLu model
def evaluate_model(trainX, trainy, testX, testy):
	# define model
	#verbose, epochs, batch_size = 0, 25, 64
	verbose, epochs, batch_size = 1, 25, 128
	n_timesteps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainy.shape[1]
	# reshape data into time steps of sub-sequences
	n_steps, n_length = 4, 32
	trainX = trainX.reshape((trainX.shape[0], n_steps, n_length, n_features))
	testX = testX.reshape((testX.shape[0], n_steps, n_length, n_features))
	# define model
	model = Sequential()
	model.add(TimeDistributed(Conv1D(filters=64, kernel_size=3, activation='relu'), input_shape=(None,n_length,n_features)))
	model.add(TimeDistributed(Conv1D(filters=64, kernel_size=3, activation='relu')))
	model.add(TimeDistributed(Dropout(0.5)))
	model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
	model.add(TimeDistributed(Flatten()))
	model.add(LSTM(100))
	model.add(Dropout(0.5))
	model.add(Dense(100, activation='relu'))
	model.add(Dense(n_outputs, activation='softmax'))
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	# fit network
	model.fit(trainX, trainy, epochs=epochs, batch_size=batch_size, verbose=verbose)
	# evaluate model
	_, accuracy = model.evaluate(testX, testy, batch_size=batch_size, verbose=0)
	return accuracy, model

# summarize scores
def summarize_results(scores):
	print(scores)
	m, s = mean(scores), std(scores)
	print('Accuracy: %.3f%% (+/-%.3f)' % (m, s))

# run an experiment
def run_experiment(repeats=10):
	# load data
	trainX, trainy, testX, testy = load_dataset()
	# repeat experiment
	scores = list()
	for r in range(repeats):
		score, kmodel = evaluate_model(trainX, trainy, testX, testy)
		score = score * 100.0
		print('>#%d: %.3f' % (r+1, score))
		scores.append(score)
	# summarize results
	summarize_results(scores)
	# convert keras model to tflite and save it
	# Save the model
	kmodel.save('kerasModel.h5')
	#converter = tf.lite.TFLiteConverter.from_keras_model(kmodel)
	#tflite_model = converter.convert()
	#tflite_model_name = "mymodel.tflite"
	#open(tflite_model_name, "wb").write(tflite_model)

	#model.save("model.h5")

# run the experiment
#run_experiment()

trainX, trainy, testX, testy = load_dataset('C:\\Brainlab\\CoughDetectionApp\\src\\')