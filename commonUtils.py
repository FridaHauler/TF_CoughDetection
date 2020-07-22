import numpy as np
from numpy import std
from numpy import mean
from numpy import dstack
import pandas as pd
from pandas import read_csv

import os, re, shutil
from os import listdir
from os.path import isfile, join
from copy import copy
import csv

import matplotlib.pyplot as plt
from matplotlib import pyplot

import tensorflow as tf
from tensorflow import keras
from keras.utils import to_categorical


#from tensorflow.keras.utils import to_categorical


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
    return loaded

# load a dataset group, such as train or test
def load_dataset_group(group, prefix=''):
    filepath = prefix + group + '/Inertial Signals/'
    # load all 9 files as a single array
    filenames = list()
    # total acceleration
    filenames += ['total_acc_x_'+group+'.txt', 'total_acc_y_'+group+'.txt', 'total_acc_z_'+group+'.txt']
    # body acceleration
    filenames += ['body_acc_x_'+group+'.txt', 'body_acc_y_'+group+'.txt', 'body_acc_z_'+group+'.txt']
    # body gyroscope
    filenames += ['body_gyro_x_'+group+'.txt', 'body_gyro_y_'+group+'.txt', 'body_gyro_z_'+group+'.txt']
    # load input data
    X = load_group(filenames, filepath)
    # load class output
    y = load_file(prefix + group + '/y_'+group+'.txt')
    print(type(X), 'and the type of the y group is:', type(y))
    return X, y

# load the dataset, returns train and test X and y elements
def load_dataset(prefix=''):
    # load all train
    trainX, trainy = load_dataset_group('train', prefix + 'HARDataset/')
    print('x train shape:', trainX.shape, 'train y shape:',  trainy.shape)
    # load all test
    testX, testy = load_dataset_group('test', prefix + 'HARDataset/')
    print(testX.shape, testy.shape)
    # zero-offset class values
    trainy = trainy - 1
    testy = testy - 1
    # one hot encode y
    trainy = to_categorical(trainy)
    testy = to_categorical(testy)
    print("trainX.shape: ", trainX.shape, 'trainy.shape: ', trainy.shape, 'and the test: ', testX.shape, testy.shape)
    print('__________**********_________', type(trainX))
    return trainX, trainy, testX, testy


def loadFile2List(filepath):
    dataframe = read_csv(filepath)
    return dataframe.values.tolist()

def getFilesFromDir(folder,prefix):
    fileRegex = re.compile(r'([a-zA-Z0-9_])%s\.csv' % prefix)
    folder = os.path.abspath(folder)
    # FileNotFoundError: [Errno 2] No such file or directory:   '*_<prefix>.csv'
    os.chdir(folder)  # This line is to avoid the FileNotFoundError.
    # Make a list to contain the file with prefix.
    fileNames = list()
    for filename in os.listdir(folder):
        if re.search(fileRegex, filename):
            fileNames.append(filename)
    return fileNames

def shuffle_data(trans_master_record, backup_record, trunc_master_label, master_file_list):
    length = len(trans_master_record)
    x = [i for i in range(0, length)]
    print('xxxxxxxxx10xxxxxxxxxxx: ', x[0:10])
    np.random.seed()
    np.random.shuffle(x)
    new_trans_master_record = []
    new_backup_record = []
    new_trunc_master_label = []
    new_master_file_list = []
    for i in x:
        new_trans_master_record.append(copy(trans_master_record[i]))  
        new_backup_record.append(copy(backup_record[i]))
        new_trunc_master_label.append(copy(trunc_master_label[i]))
        new_master_file_list.append(copy(master_file_list[i]))
    return new_trans_master_record, new_backup_record, new_trunc_master_label, new_master_file_list

def readAndConcatCoughFrames(mypath):
    allfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    
    master_record = []
    master_label = []
    master_file_index = {}
    master_file_list = []
    record_data_lengths = []
    data_records=[]
    #try:
    #    os.remove("log.txt") 
    #except OSError:
    #    pass
    #log = open("log.txt", "w+")
    rand_state = np.random.RandomState(0)

    index = 0
    
    allfiles = list(set(allfiles)) 
    #print('*********fileParsed********', allfiles)
    for file in allfiles:
        master_file_list.append(file)
        file_addr = mypath + file
        master_file_index[file] = index
        with open(file_addr,'r') as fp:
            read_file = fp.readlines()[6:]
            #print('[debug]*****lenght of the file*****', len(read_file), read_file[0])
            record_data_lengths.append(len(read_file))
            for i in range(0, len(read_file)):
                    read_file[i] = read_file[i].rstrip()

        new_read_file = []
        for item in read_file:
            new_read_file.append(item.split(','))
        #print('###################', new_read_file)
        reading = []
        label = []
        
        for item in new_read_file:
            if item[0] != 'noLabel' and rand_state.rand()<0.01:
                    #print('what to do with the noLabels?')
                    label.append(item[0])
                    reading.append(item[1:])
            elif rand_state.rand()<0.33:
                   #print('what to do with the noLabels?')
                    label.append(item[0])
                    reading.append(item[1:])
        master_record.append(reading)
        master_label.append(label)
        print('test format:', len(reading), len(master_record))
        #data_records = split_list_into_chunks(master_record, chunk_size=6)     
        
        index = index + 1
        #check and find if any readings have one or more labels missing
    print("[debug]: nr of read files...", index,'lenght of the data read (master_record): ',  len(master_record), '_and the chunked data len: ', len(data_records))
    
    #print("Creating model training and testing data-blocks from master file list:...", master_file_list[0:10])
  

    #65-35 split
    #train_record = split_list_into_chunks(master_record, chunk_size=6)  
    train_label = master_label[0:round(len(master_label)/2)]
    train_record = master_file_list[0:round(len(master_record)/2)]
    print('[debug]: nr of trained label entries: ', len(master_record) )
    
    test_record = master_record[round(len(master_record)/2) : ]
    test_label = master_label[round(len(master_label)/2): ]
    test_files = master_file_list[round(len(master_file_list)/2):  ]
    
    #num_iterations = 1
    #frames = np.reshape(train_record, (1,6))
    #pd.DataFrame(data=test_record, index='1', columns='frames*', dtype=None)
    print('______________________let us see', type(train_label), type(test_record))
    aTrain_label = np.asarray(train_label, dtype=np.float32)
    aTrain_data = np.asarray(train_record, dtype=np.float32)
    aTest_label = np.asarray(test_label, dtype=np.float32)
    aTest_data = np.asarray(test_record, dtype=np.float32)
    return aTrain_label, aTrain_data, aTest_label, aTest_data


def split_list_into_chunks(data, chunk_size=6):
    """
    Accepts a list and splits it into even-sized chunks (defaulted to 6). Assumes len(data) is divisible by chunk_size.

    :param data:
    :type data: list
    :param chunk_size:
    :type chunk_size: int
    :return: list
    """
    return [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]


def reshape(df_file, n_steps, n_length):
        #n_steps, n_length = 50, 301
        #trainX = df_file.reshape((df_file.shape[0], n_steps, n_length))
        start = 0
        for i in range(0, len(df_file.index)):
           if (i + 1)%50 == 0:
                    result = df_file.iloc[start:i+1].values.reshape(n_steps,n_length)
                    start = i + 1
        return (result)

def writeList2csv(data, csvFileName):
    csvFile = open(csvFileName, 'wb')
    with csvFile as csvF:
        csvWriter = csv.writer(csvF)
        for item in data:
            print(item)
            csvWriter.writerow(item[1:])
        print(item)
    csvFile.close()
    return
        
mode = 'debug'        
if __name__ == "__main__":
    if mode == 'debug':
        listAllFiles = list()
        filepath = "C:\\Brainlab\\CoughDetectionApp\\src\\tmp\\train\\"
        #filepath ='\\\destore\\RDData\\Surgery\\Cough\\Frames50\\'
        #filepath = "c:\\Users\\frida.hauler\\Anaconda3\\myTries\\dataSets\\"
        print(getFilesFromDir(filepath, 'frames'))
        train_data, train_labels, test_data, test_labels = readAndConcatCoughFrames(filepath)
        
        train_data = train_data.reshape(-1, 50, 6, order='F')
        train_labels =train_labels.reshape(-1,50,6,order='F')
        test_data = test_data.reshape(-1,50,6,order='F')
        test_labels = test_labels.reshape(-1,50,6,order='F')

        print('[debug] lenght of the train/test data and labels: ', len(train_data),': ', len(train_labels), ' ', len(test_data), ' ', len(test_labels))
        
    elif (mode == 'HAR'):
        trainX, trainy, testX, testy = load_dataset('C:\\Brainlab\\CoughDetectionApp\\src\\')
    else:
        df_file = pd.read_csv("C:\\Brainlab\\CoughDetectionApp\\src\\tmp\\train\\iffw9UfadVxlxHZ53fyE_frames.csv", header=None)
        print(df_file, len(df_file))
        result=[]
        n_steps, n_length = 50, 301
        #trainX = df_file.reshape((df_file.shape[0], n_steps, n_length))
        start = 0
        for i in range(0, len(df_file.index)):
           if (i + 1)%50 == 0:
                    result = df_file.iloc[start:i+1].values.reshape(n_steps,n_length)
                    start = i + 1
           print((result))
