import numpy as np
f = open('./data/training_data100','r')

f_training = open('./data/pre_pytorch/training_data100','w')
f_vailding = open('./data/pre_pytorch/vailding_data100','w')

data = f.readlines()
print(len(data))

idx = np.zeros(len(data), dtype='bool')
idx[np.random.choice(len(data), size=10000, replace=False).astype('int64')] = True

for index,i in enumerate(idx):
    if(i):
        f_vailding.write(data[index])
    else:
        f_training.write(data[index])
del data,idx

f.close()
f_training.close()
f_vailding.close()
