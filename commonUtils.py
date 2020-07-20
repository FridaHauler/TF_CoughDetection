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
def load_HARdataset_group(group, prefix=''):
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
def load_HARdataset(prefix=''):
    # load all train
    trainX, trainy = load_HARdataset_group('train', prefix + 'HARDataset/')
    print('x train shape:', trainX.shape, 'train y shape:',  trainy.shape)
    # load all test
    testX, testy = load_HARdataset_group('test', prefix + 'HARDataset/')
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

def readfiles(mypath):
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    
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
    
    #removing files which are known to have missing t_peak, t_end or s_end 
    missing_label_files = \
    ['yyJ2vgHtQJuhMsdhAx41_frames.csv'] 
    
    onlyfiles = list(set(onlyfiles) - set(missing_label_files)) 
    print('*********fileParsed********', onlyfiles)
    for file in onlyfiles:
        master_file_list.append(file)
        file_addr = mypath + file
        master_file_index[file] = index
        with open(file_addr,'r') as fp:
            read_file = fp.readlines()[6:]
            print('*****lenght of the file*****', len(read_file))
            record_data_lengths.append(len(read_file))
            for i in range(0, len(read_file)):
                    read_file[i] = read_file[i].rstrip()
        
        new_read_file = []
        for item in read_file:
            new_read_file.append(item.split(','))
        #print('###################', item.split(','))
        reading = []
        label = []
        
        for item in new_read_file:
         
            label.append(item[0])
            reading.append(item[1:50])
            #print('item:', item[0:3])

        master_record.append(reading)
        master_label.append(label)
        
        index = index + 1
    
    print("Creating the training and testing data-blocks for model input...")
        
    trans_master_record, backup_record, trunc_master_label, master_file_list = \
            shuffle_data(trans_master_record, backup_record, trunc_master_label, master_file_list)

    #65-35 split
    train_record = trans_master_record[0:50]
    print(train_record)
    train_label = trunc_master_label[0:50]
    train_files = master_file_list[0: 50]
    
    test_record = trans_master_record[50:122]
    test_label = trunc_master_label[50:122]
    test_files = master_file_list[50: 122]
    orig_test_record = backup_record[50:122]
    
    num_iterations = 1
    
    print("Plotting results...")
    ret = 1
    ret = new_plot1(train_record, test_record, orig_test_record,\
        test_label, test_files, num_iterations)
    if ret == 0:
        print("Plotting function returned 0 - one of the labels not in predictions...")
    else:
        print("Plotting completed successfully")
    
mode = 'debug'        
if __name__ == "__main__":
    if mode == 'debug':
        listAllFiles = list()
        #filepath = "C:\\Brainlab\\CoughDetectionApp\\src\\tmp\\"
        filepath ='\\\destore\\RDData\\Surgery\\Cough\\Frames50\\'
        #filepath = "c:\\Users\\frida.hauler\\Anaconda3\\myTries\\dataSets\\"
        print(getFilesFromDir(filepath, 'frames'))
        readfiles(filepath)
    else:
        trainX, trainy, testX, testy = load_HARdataset('C:\\Brainlab\\CoughDetectionApp\\src\\')