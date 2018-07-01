import argparse
from datasets.testDataset import testDataset,ToTensor,Sample
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms,utils
from evaluate import eval
import torch
from model.VAE import VAE
import torch.nn.functional as F
import numpy as np
import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def main(args):
    if(args.loadModel == "" or args.output==""):
        print('please input loadModel and output place')
        sys.exit(-1)

    size = args.size
    if(size == 100):
        n_items = 245614
    else:
        n_items = 220915
    
    p_dims = [300, 800, n_items]
    
    print('loading data')
    test_dataset = testDataset( file_name= args.loadData,
                                           transform=transforms.Compose([
                                               Sample(output_size=size,n_items=n_items),
                                               ToTensor()]))
    dataloader = DataLoader(test_dataset,batch_size=args.batchSize,num_workers=args.num_threads)

    print('building model')
    vae = VAE(p_dims=p_dims).cpu()
    if(args.loadModel != ""):
        print('loading from previous model')
        vae.load_state_dict(torch.load(args.loadModel))
    vae = vae.cuda()
    print(vae)    


    print('start testing model')
    f = open(args.output,'w')
    with torch.no_grad():
        for _,sample_val in enumerate(dataloader):
            sample_val['seed'] = sample_val['seed'].cuda()
            output_val,_,_ = vae(sample_val['seed'],training=False)
            
            seed = sample_val['seed'].cpu().detach().numpy()
            output_val = output_val.cpu().detach().numpy()

            output_val[seed==1] = -np.inf

            eval.output_ans(output_val,sample_val['pid'],f)
            
                        

if(__name__ == '__main__'):
    parser = argparse.ArgumentParser(description='Process some integers.')
    
    parser.add_argument('--batchSize', type=int, default=8, help='input batch size')
    parser.add_argument('--num_threads', type=int, default=8, help='# threads for loading data')
    parser.add_argument('--loadData', type=str, default="./data/pre_pytorch/", help='place to load data')
    parser.add_argument('--size', type=int, default=25, help='seed song size')
    
    parser.add_argument('--loadModel',type=str, default="", help='place to load model')
    parser.add_argument('--output', type=str, default="", help='place to load model')
        
    args = parser.parse_args()
    main(args)
