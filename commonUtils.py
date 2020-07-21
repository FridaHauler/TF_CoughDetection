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

import matplotlib.pyplot as plt
from matplotlib import pyplot

from tensorflow.keras.utils import to_categorical


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

def readfiles(mypath):
    allfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    
    master_record = []
    master_label = []
    master_file_index = {}
    master_file_list = []
    record_data_lengths = []
    try:
        os.remove("log.txt") 
    except OSError:
        pass
    log = open("log.txt", "w+")
    
    index = 0
    
    allfiles = list(set(allfiles)) 
    #print('*********fileParsed********', allfiles)
    for file in allfiles:
        master_file_list.append(file)
        file_addr = mypath + file
        master_file_index[file] = index
        with open(file_addr,'r') as fp:
            read_file = fp.readlines()[6:]
            #print('*****lenght of the file*****', len(read_file))
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
            if item[0] != 'noLabel':
                    #print('what to do with the noLabels?')
                    label.append(item[0])
                    reading.append(item[1:50])

        master_record.append(reading)
        master_label.append(label)
        print('test format:', len(master_label), type(master_record))
        
        index = index + 1
        #check and find if any readings have one or more labels missing
    #print("read files...", index)
    
    #print("Creating model training and testing data-blocks from master file list:...", master_file_list[0:10])
  

    #65-35 split
    train_record = master_record[1:50]
    print(type(train_record))
    train_label = master_label[0:50]
    train_files = master_file_list[0: 50]
    
    test_record = master_record[50:100]
    test_label = master_label[50:100]
    test_files = master_file_list[50: 100]
    
    num_iterations = 1
    #frames = np.reshape(train_record, (1,6))
    print("Plotting results..." ,len(train_record))
    #pd.DataFrame(data=test_record, index='1', columns='frames*', dtype=None)
        
def reshape(df_file, n_steps, n_length):
        #n_steps, n_length = 50, 301
        #trainX = df_file.reshape((df_file.shape[0], n_steps, n_length))
        start = 0
        for i in range(0, len(df_file.index)):
           if (i + 1)%50 == 0:
                    result = df_file.iloc[start:i+1].values.reshape(n_steps,n_length)
                    start = i + 1
        return (result)
        
mode = 'debug'        
if __name__ == "__main__":
    if mode == 'debug':
        listAllFiles = list()
        filepath = "C:\\Brainlab\\CoughDetectionApp\\src\\tmp\\train\\"
        #filepath ='\\\destore\\RDData\\Surgery\\Cough\\Frames50\\'
        #filepath = "c:\\Users\\frida.hauler\\Anaconda3\\myTries\\dataSets\\"
        print(getFilesFromDir(filepath, 'frames'))
        readfiles(filepath)
    elif (mode == 'HAR'):
        trainX, trainy, testX, testy = load_HARdataset('C:\\Brainlab\\CoughDetectionApp\\src\\')
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
