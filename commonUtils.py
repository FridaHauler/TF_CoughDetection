import numpy as np, numpy
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
    
    # load all test
    testX, testy = load_dataset_group('test', prefix + 'HARDataset/')
    print(type(testX), '_____testX, testy_______', type(testy))
    # zero-offset class values
    trainy = trainy - 1
    testy = testy - 1
    # one hot encode y
    trainy = to_categorical(trainy)
    testy = to_categorical(testy)
    #print("trainX.shape: ", trainX.shape, 'trainy.shape: ', trainy.shape, 'and the test: ', testX.shape, testy.shape)
    
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
        
        reading = []
        label = []
        
        for item in new_read_file:
            if item[0] != 'noLabel' and rand_state.rand()<0.33:
                    #print('labels?', item[0])
                    label.append(item[0])
                    reading.append(item[1:])
            elif rand_state.rand()<0.001:
                   #print('what to do with the noLabels?', item[0], item[1:])
                    label.append(item[0])
                    reading.append(item[1:])
            master_record.append(reading)
            master_label.append(label)
        
        #data_records = split_list_into_chunks(master_record, chunk_size=6)     
        
        index = index + 1
        #check and find if any readings have one or more labels missing
    # endfor: parsing all the data from the folder to master_record(1-300:) and master_label(0:)
    print("[debug]: total record: ",len(master_record) )
    print("[debug]: total record: " ,np.array(master_record, dtype=np.float).shape)
    
    #train_record = split_list_into_chunks(master_record, chunk_size=6)  
    N=len(master_record)/2
    
    train_label = master_label[0:round(N)]
    train_record = master_record[0:round(len(master_record)/2)]
    
    test_label = master_label[round(len(master_label)/2): ]
    test_record = master_record[round(len(master_record)/2) : ]
    
    test_files = master_file_list[: ]
    print('[debug]: from : ', len(test_files), 'files, records split to training: ',  len(train_label), len(train_record), 'and to test: ', len(test_record) )
    
    #num_iterations = 1
    #frames = np.reshape(train_record, (1,6))
    #pd.DataFrame(data=test_record, index='1', columns='frames*', dtype=float32)
    
    print('train_label type (expected would be a numpy.array____): ', len(train_label), len(train_record), 'len(test_record): ', len(test_record))
    
    for x in train_record:
        print(len(x))

    train_record = np.array(train_record, dtype=np.float)
    train_label = np.array(train_label, dtype=np.float)
    test_record = np.array(test_record, dtype=np.float)
    test_label = np.array(test_label, dtype=np.float)

    print('All data before reshape:')
    print(len(train_label), '_________', train_label)
    print(len(train_record), '$$$$$$$$$$$$$$$', train_record)
    print(test_label)
    print(test_record)

    train_record = train_record.reshape(-1, 50, 6, order='F')
    test_record = test_record.reshape(-1,50,6,order='F')

    print('All data:')
    print(train_label)
    print(train_record)
    print(test_label)
    print(test_record)

    return train_record, train_label, test_record, test_label


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
            #print(item)
            csvWriter.writerow(item[1:])
        #print(item)
    csvFile.close()
    return

def readAll2PD(fileLoc):
    import glob, math
    df_file = pd.concat(map(pd.read_csv, glob.glob(os.path.join(fileLoc, "*.csv"))))
    #df_file = pd.read_csv("C:\\Brainlab\\CoughDetectionApp\\src\\tmp\\train\\iffw9UfadVxlxHZ53fyE_frames.csv", header=None)
    print(df_file, df_file.shape)

    total_size=len(df_file)
    train_size=math.floor(0.66*total_size) #(2/3 part of my dataset)
    #training dataset
    train_data=df_file.head(train_size)
    #test dataset
    test_data=df_file.tail(len(df_file) -train_size)

    train_label = train_data.iloc[:, 0]
    train_label = to_categorical(train_label)
    label_map = {"noLabels": 0, "Single cough": 1, "Multiple coughs": 2, "Clear throat": 3}

    train_label = train_label[0].map(label_map)
    print(train_label_nr)

    train_data = train_data.iloc[:,1:]
    train_data = np.array(train_data).reshape(-1, 50, 6, order='F')
  

    test_label = test_data.iloc[:,0]
    test_label = to_categorical(test_label)

    test_data = test_data.iloc[:,1:]
    test_data = np.array(test_data).reshape(-1, 50, 6, order='F')

    '''debug:
    print('All data:')
    print(train_label)
    print(train_data)
    print(test_label)
    print(test_data)

    result=[]
    n_steps, n_length = 50, 301
    #trainX = df_file.reshape((df_file.shape[0], n_steps, n_length))
    start = 0
    for i in range(0, len(df_file.index)):
        if (i + 1)%50 == 0:
            result = df_file.iloc[start:i+1].values.reshape(n_steps,n_length)
            start = i + 1
    
    N= round(len(result)/2)
    train_label = result[0:round(len(result)/2)]
    train_data  = result[1:N]
    test_label = result[N+1:]
    test_data = result[N+1:]
    '''
    return train_data, train_label, test_data, test_label
        
mode = 'dataframeBased'        
if __name__ == "__main__":
    if mode == 'debug':
        print('##############debug mode for cough detection data ############## ')
        listAllFiles = list()
        filepath = "C:\\Brainlab\\CoughDetectionApp\\src\\tmp\\"
        #filepath ='\\\destore\\RDData\\Surgery\\Cough\\Frames50\\'
        #filepath = "c:\\Users\\frida.hauler\\Anaconda3\\myTries\\dataSets\\"
        print(getFilesFromDir(filepath, 'frames'))
        train_data, train_labels, test_data, test_labels = readAndConcatCoughFrames(filepath)
        
       

        print('[debug] lenght of the train/test data and labels: ', len(train_data),': ', len(train_labels), ' ', len(test_data), ' ', len(test_labels))
        
    elif (mode == 'HAR'):
        print('_______________HAR data__________')
        trainX, trainy, testX, testy = load_dataset('C:\\Brainlab\\CoughDetectionApp\\src\\')
    else:
        '''
        reshape the csv to (*,50,6) with dataframes
        '''
    readAll2PD('C:\\Brainlab\\CoughDetectionApp\\src\\tmp')


