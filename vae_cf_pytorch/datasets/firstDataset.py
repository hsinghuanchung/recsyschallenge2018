from torch.utils.data import Dataset,DataLoader
import numpy as np
import torch


class itemDataset(Dataset):
    def __init__(self, file_name,transform=None):
        self.file = open(file_name,'r')
        self.user_data =[]
        
        for line in self.file:
            line = line.strip().split()
            if(len(line)==100):
                continue
            arr = []
            for data in line:
                arr.append(int(data))
            self.user_data.append(arr)
        
        self.file.close()
        self.transform = transform
    def __len__(self):
        return len(self.user_data)
    def __getitem__(self, idx):
        sample = self.user_data[idx].copy()
        
        if self.transform:
            sample = self.transform(sample)
        return sample

class Sample(object):
    def __init__(self,output_size,n_items):
        assert(isinstance(output_size,(int)))
        assert(isinstance(n_items,(int)))
        
        self.output_size = output_size
        self.n_items = n_items
        
        np.random.seed(98765)
        
    def __call__(self,sample):
        seed_arr,ans_arr=[],[]
        
        n_items_u = len(sample)
		
        for index in range(len(sample)):
            if(sample[index]==-1):
                continue
            
            if(index < self.output_size):
                seed_arr.append(sample[index])
            ans_arr.append(sample[index])
        
        seed_np = self.convert_data(seed_arr)
        ans_np = self.convert_data(ans_arr)
        
        return {'seed': seed_np , 'total': ans_np,'length': np.array([n_items_u])}

    def convert_data(self,arr):
        data = np.zeros([self.n_items],dtype='float32')
        for i in arr:
            data[i] = 1
        return data

class ToTensor(object):
    def __call__(self,sample):
        return{
            'seed':torch.from_numpy(sample['seed']),
            'total':torch.from_numpy(sample['total']),
            'length':torch.from_numpy(sample['length'])}
