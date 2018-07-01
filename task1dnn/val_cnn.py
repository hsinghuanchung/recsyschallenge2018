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
os.environ["CUDA_VISIBLE_DEVICES"]="0"
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
set_session(sess)
reverse_tk = pickle.load(open('/mnt/data/b04901058/recsys/reverse_token0_Xfull.pkl','rb'))
def load_partition(val_ratio=0.1):
    total_num = len(os.listdir('/mnt/data/b04901058/recsys/0_Y'))
    list_ID = np.arange(total_num)
    
    return {'train':list_ID[int(np.floor(total_num*val_ratio)):],'validation':np.arange(20000)}

def top500(predict,val_data,resultfile):
    with open(resultfile,'w') as f:
        for idx, row in enumerate(predict):
            print(idx)
            rank = np.argsort(-row)
            top500 = checkduplicate(rank,val_data[idx,:])
            for i in top500:
                f.write(i)
                f.write(' ')
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


def main():

    modelname1 = argv[1]
    #modelname2 = argv[2]
    resultfile = argv[2]
    
    val_data = []
    print('load val data')
    for i in range(20000):
        val_data.append(np.load('/mnt/data/b04901058/recsys/0_X/' + str(i) + '.npy'))
    val_data = np.stack(val_data)
    print(val_data.shape)
    print(val_data[0])
    
    cnn_model1 = load_model(modelname1)
    #cnn_model2 = load_model(modelname2)
    val_predict1 = cnn_model1.predict(val_data,verbose=1) 
    #val_predict2 = cnn_model2.predict(val_data,verbose=1) 
    #val_predict = (val_predict1 + val_predict2) / 2 
    top500(val_predict1,val_data,resultfile)
    
    print(val_predict.shape) 
if __name__ == '__main__':
    main()
