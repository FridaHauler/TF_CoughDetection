from numpy import mean
from numpy import std
from numpy import dstack
from pandas import read_csv
'''Win 7 
'''
from keras.models import Sequential, save_model
from keras.layers import Dense, InputLayer
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import TimeDistributed
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D

'''Win 10
os.environ['PATH'] = 'C:\\cuDNN\\cudnn-10.1-windows10-x64-v7.6.5.32\\cuda\\bin;' \
	'C:\\Program Files\\NVIDIA GPU Computing Toolkit\CUDA\\v10.1\\bin;\{}'.format(os.environ['PATH'])

from tensorflow.keras.models import Sequential, save_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import MaxPooling1D

'''


from keras.utils import to_categorical
from matplotlib import pyplot
from tensorflow.keras import Model
import tensorflow as tf
from tensorflow import lite

import os

from tensorflow.keras import datasets, layers, models
from commonUtils import load_file, load_group, load_dataset_group, load_dataset, readAndConcatCoughFrames, readAll2PD

# fit and evaluate a LSTM ReLu model
def evaluate_model(trainX, trainy, testX, testy):
	# define model
	print(trainX.shape[2], trainy.shape, testX.shape, testX.shape)
	n_timesteps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainy.shape[1]
	
	verbose, epochs, batch_size = 0, 2, 4
	#verbose, epochs, batch_size = 1, 25, 128
	#n_timesteps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainy.shape[1]
	# reshape data into time steps of sub-sequences
	n_steps, n_length = 6, 50
	
	trainX = trainX.reshape((trainX.shape[0], n_steps, n_length, n_features))
	testX = testX.reshape((testX.shape[0], n_steps, n_length, n_features))

	#trainy = trainX[trainX.columns[0]]
	#testX = testX.shape[1]
		
	print('#####################')
	print(trainX.shape)
	print(trainy.shape)
	print('#####################')

	print('########tests#############')
	print(testX.shape)
	print(testy.shape)
	print('#####################')

	
	# define model
	model = Sequential()
	model.add(InputLayer(input_shape=(n_timesteps, n_features)))
	model.add(Conv1D(filters=32, kernel_size=3, activation='relu'))
	model.add(Conv1D(filters=16, kernel_size=3, activation='relu'))
	model.add(Dropout(0.5))
	model.add(MaxPooling1D(pool_size=2))
	# model.add(Flatten())
	model.add(LSTM(100))
	model.add(Dropout(0.5))
	#model.add(Dense(16, activation='relu'))
	#model.add(Dense(n_outputs, activation='softmax'))
	model.add(Dense(100, activation='relu'))
	model.add(Dense(n_outputs, activation='softmax'))
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	model.summary()
	# fit network
	model.fit(trainX, trainy, epochs=epochs, batch_size=batch_size, verbose=verbose)
	_, accuracy = model.evaluate(testX, testy, batch_size=batch_size, verbose=0)
	

	'''
	# define model
	model = Sequential()
	#model.add(TimeDistributed(Conv1D(filters=64, kernel_size=3, activation='relu'), input_shape=(None,n_length,n_features)))
	model.add(TimeDistributed(Conv1D(filters=64, kernel_size=3, activation='relu'), input_shape=(batch_size,n_length,n_features)))
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
	trainy = scaler.inverse_transform([trainy])
	# evaluate model
	_, accuracy = model.evaluate(testX, testy, batch_size=batch_size, verbose=0)
	model.summary()
	'''
	return accuracy, model

# summarize scores
def summarize_results(scores):
	print(scores)
	m, s = mean(scores), std(scores)
	print('Accuracy: %.3f%% (+/-%.3f)' % (m, s))

def export_tflite(classifier):
	with tf.compat.v1.Session() as sess:
		# First let's load meta graph and restore weights
		latest_checkpoint_path = classifier.latest_checkpoint()
		saver = tf.train.import_meta_graph(latest_checkpoint_path + '.meta')
		saver.restore(sess, latest_checkpoint_path)
		
		# Get the input and output tensors
		input_tensor = sess.graph.get_tensor_by_name("dnn/input_from_feature_columns/input_layer/concat:0")
		out_tensor = sess.graph.get_tensor_by_name("dnn/logits/BiasAdd:0")

		# here the code differs from the toco example above
		#sess= tf.compat.v1.Session()
		sess.run(tf.global_variables_initializer())
		converter = tf.lite.TFLiteConverter.from_session(sess, [input_tensor], [out_tensor])
		tflite_model = converter.convert()
		open("converted_model.tflite", "wb").write(tflite_model)

# run an experiment
def run_experiment(repeats=2):
	# load data
	#print('___________current directory:_____________', os.path.realpath('.'))
	#path_dataset = '\\\destore\\RDData\\Surgery\\Cough\\Frames50\\'
	#trainX, trainy, testX, testy = load_dataset(path_dataset)

	path_CoughDataset = 'C:\\Brainlab\\CoughDetectionApp\\src\\tmp\\'
	#trainX, trainy, testX, testy = readAndConcatCoughFrames(path_CoughDataset)
	'''
	trainX, trainy, testX, testy  = train_data, train_label, test_data, test_label 
	'''
	trainX, trainy, testX, testy = readAll2PD(path_CoughDataset)
		
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
	# Save the model
	kmodel.save('kerasModel.h5')

	converter = tf.lite.TFLiteConverter.from_keras_model(kmodel)
	# NEED THIS
	converter.experimental_new_converter = True

	tflite_model = converter.convert()
	tflite_model_name = "mymodel.tflite"
	open(tflite_model_name, "wb").write(tflite_model)

	#model.save("model.h5")

# run the experiment
run_experiment()
