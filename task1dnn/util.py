import numpy as np
import keras
import pickle
import os
import gensim
import json
import re
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
def load_dict(dict_name):
    with open(dict_name,'rb') as f:
        return pickle.load(f)

def normalize_name(name):
    name = name.lower()
    name = re.sub(r"[.,\/#!$%\^\*;:{}=\_`~()@]", ' ', name)
    name = re.sub(r'\s+', ' ', name).strip()
    return name
#bin_vec = '/home/yuchichen/word2vec/trunk/vectorsA_20_100.bin'
#w2vmodel = gensim.models.KeyedVectors.load_word2vec_format(bin_vec, binary=True)
#tk = pickle.load(open('/mnt/data/b04901058/recsys/token0_Xfull.pk','rb'))

class DataManager:
    def __init__(self):
        self.data = {}

    def add_data(self,name,data_path):
        X = []
        if name == 'seed':
            print('read seed data')
            with open(data_path,'r') as f:
                for line in f:
                    line = line.strip().split(',')
                    X.append(line[0])
        elif name == 'truth':
            print('read truth data')
            with open(data_path,'r') as f:
                for line in f:
                    line = line.strip().split(',')
                    X.append(line[1])
        elif name == 'test1':
            print('read test task1 input')
            with open(data_path,'r') as reader:
                jf = json.loads(reader.read())
                for i in range(1000):
                    title = normalize_name(jf["playlists"][i]["name"])
                    X.append(title)
                print(jf["playlists"][500]) 
                print(jf["playlists"][999])
                print(X)
        self.data[name] = [X]
    
    def tokenize(self, vocab_size):
        print('create new tokenizer')
        self.tokenizer = Tokenizer(num_words=vocab_size,lower=False,oov_token='¬',filters='')
        
        for key in self.data:
            print('fit on %s'%key)
            texts = self.data[key][0]
            self.tokenizer.fit_on_texts(texts)
        
        self.tokenizer.word_index = {e:i for e,i in self.tokenizer.word_index.items() if i <= vocab_size} 
        self.tokenizer.word_index[self.tokenizer.oov_token] = vocab_size + 1
    

    def save_tokenizer(self, path):
        print('save tokenizer to %s'%path)
        pickle.dump(self.tokenizer, open(path, 'wb'))


    def load_tokenizer(self, path):
        print('load tokenizer from %s'%path)
        self.tokenizer = pickle.load(open(path, 'rb'))
    

    def to_sequence(self, maxlen):
        self.maxlen = maxlen
        for key in self.data:
            if key == 'seed' or key == 'test1':
                print ('converting %s to sequences'%key)
                tmp = self.tokenizer.texts_to_sequences(self.data[key][0])
                self.data[key][0] = np.array(pad_sequences(tmp, maxlen=maxlen))
    
    def tosave_label(self,path):
        for key in self.data:
            if key == 'truth':
                print ('converting %s to label'%key)
                for idx, truth in enumerate(self.data[key][0]):
                    y_label = []
                    print('saving the %s label'%idx)
                    truth = truth.strip().split()
                    for t in truth:
                        if t in self.tokenizer.word_index:
                            y_label.append(self.tokenizer.word_index[t])
                    final_path = os.path.join(path,str(idx)+'.pkl')
                    pickle.dump(y_label,open(final_path,'wb'))
                    

    def embedding_matrix(self):
        print('making embedding matrix')
        word_index = self.tokenizer.word_index
        embedding_matrix = np.zeros((len(word_index)+1,200))
        bin_vec = '/home/yuchichen/word2vec/trunk/vectorsA.bin'
        w2vmodel = gensim.models.KeyedVectors.load_word2vec_format(bin_vec, binary=True)
        for word, i in word_index.items():
            if word in w2vmodel:
                embedding_vector = w2vmodel[word]
                embedding_matrix[i] = embedding_vector
            else: continue
        return word_index, embedding_matrix

    def save_sequence(self,path):
        for key in self.data:
            if key == 'seed' or key == 'test1':
                for matrix in self.data[key]:
                    for idx,row in enumerate(matrix):
                        print('saving ' + key + ': ' + str(idx))
                        np.save(path+str(idx),row)


class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, batch_size=32,shuffle=True):
        'Initialization'
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.n_classes = 230001 
        self.dim = 200
        self.shuffle = shuffle
        self.on_epoch_end()


    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        #print(index)
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
    
    def to_multilabel(self, y, num_classes=None):
        if num_classes is None: num_classes = self.n_classes
        multilabel = np.zeros((len(y),num_classes))
        for idx, l in enumerate(y):
            for track in l:
                multilabel[idx,track-1] = 1  # because the word index of tokenizer starts from 1
        
        return multilabel
    

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = []
        y = []
        
        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            #with open('/mnt/data/b04901058/recsys/25_X/' + str(ID) + '.pkl','rb') as f:
            #    X.append(pickle.load(f))
            # Store class
            X.append(np.load('/mnt/data/b04901058/recsys/0_X/' + str(ID) + '.npy'))
            with open('/mnt/data/b04901058/recsys/0_Y/' + str(ID) + '.pkl','rb') as f:
                y.append(pickle.load(f))
        X = np.stack(X)
        #print('input seeds: '+str(X.shape))
        return X, self.to_multilabel(y)
