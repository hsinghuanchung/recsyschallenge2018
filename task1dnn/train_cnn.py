import numpy as np
from sys import argv
import os
from util import DataGenerator
from util import DataManager
import pickle
from keras.models import Sequential, Model
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import Input,Conv1D, Conv2D, MaxPooling1D,AveragePooling1D, Flatten, AveragePooling2D, GlobalAveragePooling1D
from keras.layers.embeddings import Embedding
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD, Adam
from keras.utils import np_utils, plot_model
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from keras.layers.merge import concatenate
os.environ["CUDA_VISIBLE_DEVICES"]="0"
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
set_session(sess)
TRACK_DICT_SIZE = 230001 
def cnn25(word_index,embedding_matrix):
    model = Sequential()
    e = Embedding(len(word_index)+1,200,weights=[embedding_matrix],trainable=False)
    model.add(e)
    model.add(Conv1D(32, kernel_size=3, activation='relu', padding='same'))
    model.add(BatchNormalization(axis = -1))
    model.add(AveragePooling1D(2))
    # 32 * (12)
    model.add(Conv1D(64, kernel_size=3, activation='relu', padding='same'))
    model.add(BatchNormalization(axis = -1))
    model.add(AveragePooling1D(2))
    # 64 * 200 * (6)
    
    model.add(Conv1D(128, kernel_size=3, activation='relu', padding='same'))
    model.add(BatchNormalization(axis = -1))
    model.add(AveragePooling1D(2))
    
    model.add(GlobalAveragePooling1D())
    
    model.add(Dense(units = TRACK_DICT_SIZE, activation = 'sigmoid'))
    model.summary()
    return model

def cnn100(word_index,embedding_matrix):
    model = Sequential()
    e = Embedding(len(word_index)+1,200,weights=[embedding_matrix],trainable=False)
    model.add(e)
    model.add(Conv1D(32, kernel_size=3, activation='relu', padding='same'))
    model.add(BatchNormalization(axis = -1))
    model.add(AveragePooling1D(2))
    # 32 * (50)
    model.add(Conv1D(64, kernel_size=3, activation='relu', padding='same'))
    model.add(BatchNormalization(axis = -1))
    model.add(AveragePooling1D(2))
    # 64 * 200 * (25)
    
    model.add(Conv1D(128, kernel_size=3, activation='relu', padding='same'))
    model.add(BatchNormalization(axis = -1))
    model.add(AveragePooling1D(2))
    # 128 * 12 
    model.add(Conv1D(256, kernel_size=3, activation='relu', padding='same'))
    model.add(BatchNormalization(axis = -1))
    model.add(AveragePooling1D(2))
    # 256 * 6
    model.add(GlobalAveragePooling1D())
    
    model.add(Dense(units = TRACK_DICT_SIZE, activation = 'sigmoid'))
    model.summary()
    return model

def cnn0(word_index,embedding_matrix):
    model = Sequential()
    e = Embedding(len(word_index)+1,200,input_length=1,weights=[embedding_matrix],trainable=False)
    model.add(e)
    model.add(Dense(units = 192, activation = 'relu'))
    #model.add(Dense(units = 160, activation = 'relu'))
    model.add(Dense(units = 128, activation = 'relu'))
    model.add(Flatten())
    model.add(Dense(units = TRACK_DICT_SIZE, activation = 'sigmoid'))
    model.summary()
    
    return model
    
def load_partition(val_ratio=0.1):
    total_num = len(os.listdir('/mnt/data/b04901058/recsys/0_Y'))
    list_ID = np.arange(total_num)
    
    return {'train':list_ID[int(np.floor(total_num*val_ratio)):],'validation':list_ID[:int(np.floor(total_num*val_ratio))]}
def main():
    params = {'batch_size':64}
    modelname = argv[1]
    #Datasets
    partition = load_partition()
    print(len(partition['train']))
    print(len(partition['validation']))
    training_generator = DataGenerator(partition['train'],**params)
    validation_generator = DataGenerator(partition['validation'],**params)
    
    dm = DataManager()
    dm.load_tokenizer('/mnt/data/b04901058/recsys/token0_Xfull.pk') 
    word_index, embedding_matrix = dm.embedding_matrix()
    cnn_model = cnn0(word_index,embedding_matrix)
    cnn_model.compile(optimizer = 'adam', loss = 'binary_crossentropy',metrics = ['accuracy'])
    
    checkpoint = [ModelCheckpoint(modelname,  # model filename
                                   monitor='val_loss', # quantity to monitor
                                   verbose=0, # verbosity - 0 or 1
                                   save_best_only= True, # The latest best model will not be overwritten
                                   mode='auto'), # The decision to overwrite model is made 
                  EarlyStopping(monitor = 'val_loss',
                                patience = 3,
                                verbose = 0)]
    cnn_model.fit_generator(generator=training_generator,
                      validation_data=validation_generator,
                      callbacks = checkpoint,
                      verbose = 1,
                      use_multiprocessing=True,
                      epochs = 12,
                      workers=3)

if __name__ == '__main__':
    main()
