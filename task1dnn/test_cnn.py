import numpy as np
from sys import argv
import os
from util import DataGenerator
from util import DataManager
import pickle
from keras.callbacks import EarlyStopping, ModelCheckpoint
import tensorflow as tf
from keras.models import load_model
from keras.backend.tensorflow_backend import set_session
import json
os.environ["CUDA_VISIBLE_DEVICES"]="1"
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
set_session(sess)
reverse_tk = pickle.load(open('/mnt/data/b04901058/recsys/reverse_token0_Xfull.pkl','rb'))

def top500(predict,val_data,pidlist,resultfile):
    with open(resultfile,'w') as f:
        for idx, row in enumerate(predict):
            print(idx)
            rank = np.argsort(-row)
            top500 = checkduplicate(rank,val_data[idx,:])
            f.write(pidlist[idx])
            for i in top500:
                f.write(',')
                f.write('spotify:track:'+i)
            f.write('\n')

def checkduplicate(rank,valrow):
    top500 = []
    cnt = 0
    VALID = True
    for r in rank:
        if cnt == 500: break
        if len(reverse_tk[r+1]) != 22: continue
        for v in valrow:
            #print('r: '+str(r+1)+' vs '+'v: '+str(v))
            if r+1 == v: 
                VALID = False
                break
            else: 
                continue
        if VALID == True:
            top500.append(reverse_tk[r+1])
            cnt += 1
        VALID = True
    return top500

def readpid(challengeset):
    pidlist = []
    with open(challengeset,'r') as reader:
        jf = json.loads(reader.read())
    
    for i in range(1000):
        pid = jf["playlists"][i]["pid"]
        pidlist.append(str(pid))
    return pidlist

def main():

    modelname1 = argv[1]
    challengeset = argv[2] 
    resultfile = argv[3]
    pidlist = readpid(challengeset)
    print(pidlist)
    test_data = []
    print('load val data')
    for i in range(len(os.listdir('/mnt/data/b04901058/recsys/test1'))):
        test_data.append(np.load('/mnt/data/b04901058/recsys/test1/' + str(i) + '.npy'))
    test_data = np.stack(test_data)
    print(test_data.shape)
    print(test_data[0])
    
    cnn_model1 = load_model(modelname1)
    test_predict1 = cnn_model1.predict(test_data,verbose=1) 
    #val_predict = (val_predict1 + val_predict2) / 2 
    top500(test_predict1,test_data,pidlist,resultfile)
    
    print(test_predict1.shape) 
if __name__ == '__main__':
    main()
