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

from  keras.utils import to_categorical
from matplotlib import pyplot
from tensorflow.keras import Model
import tensorflow as tf
from tensorflow import lite

import os

from tensorflow.keras import datasets, layers, models
from commonUtils import load_file, load_group, load_dataset_group, load_dataset

# fit and evaluate a LSTM ReLu model
def evaluate_model(trainX, trainy, testX, testy):
	# define model
	verbose, epochs, batch_size = 0, 25, 4
	#verbose, epochs, batch_size = 1, 25, 128
	n_timesteps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainy.shape[1]
	# reshape data into time steps of sub-sequences
	n_steps, n_length = 4, 32
	print('!!!!!!!!n_steps:', n_steps, 'n_timesteps: ', n_timesteps, 'n_features', n_features, 'n_outputs', n_outputs)
	trainX = trainX.reshape((trainX.shape[0], n_steps, n_length, n_features))
	testX = testX.reshape((testX.shape[0], n_steps, n_length, n_features))
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
	# evaluate model
	_, accuracy = model.evaluate(testX, testy, batch_size=batch_size, verbose=0)
	model.summary()
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
	print('________________________', os.path.realpath('.'))
	#path_dataset = '\\\destore\\RDData\\Surgery\\Cough\\Frames50\\'
	path_dataset = 'C:\\Brainlab\\CoughDetectionApp\\src\\tmp\\train\\'
	trainX, trainy, testX, testy = load_dataset(path_dataset)
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
	export_tflite(kmodel)
	converter = tf.lite.TFLiteConverter.from_keras_model(kmodel)
	tflite_model = converter.convert()
	# Save the TF Lite model.
	with tf.io.gfile.GFile('model.tflite', 'wb') as f:
		f.write(tflite_model)

	#model.save("model.h5")

# run the experiment
run_experiment()
