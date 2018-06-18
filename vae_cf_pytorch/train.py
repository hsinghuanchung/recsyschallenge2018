import argparse
from datasets.ItemDataset import itemDataset,ToTensor,Sample
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms,utils
from evaluate import eval
import torch
from model.VAE import VAE
import torch.nn.functional as F
import numpy as np
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def main(args):
    print('qq')

    
    n_items = 220915
    p_dims = [300, 800, n_items]
    
    print('loading data')
    item_dataset = itemDataset( file_name= args.loadData+'training_data25',
                                           transform=transforms.Compose([
                                               Sample(output_size=25,n_items=n_items),
                                               ToTensor()]))
    dataloader = DataLoader(item_dataset,batch_size=args.batchSize,shuffle=True,num_workers=args.num_threads)

    vaild_dataset = itemDataset( file_name= args.loadData+'vailding_data25',
                                            transform=transforms.Compose([
                                                Sample(output_size=25,n_items=n_items),
                                                ToTensor()]))
    vaildloader = DataLoader(vaild_dataset,batch_size=args.batchSize,shuffle=True,num_workers=args.num_threads)

    print('building model')
    vae = VAE(p_dims=p_dims).cpu()
    if(args.loadModel != ""):
        print('loading from previous model')
        vae.load_state_dict(torch.load(args.loadModel+'vae_params.pkl'))

    vae = vae.cuda()
    print(vae)    
    optimizer = torch.optim.Adam(vae.parameters(),lr = args.lr)

    total_anneal_steps = 200000
    anneal_cap = 0.2

    train_loss = 0

    best_ndcg = -np.inf
    best_rpre = -np.zeros(1)
    update_count  = 0

    print('start training model')
    for index_epoch in range(args.numEpoch):
        for index,sample in enumerate(dataloader):
            sample['seed'] = sample['seed'].cuda()
            sample['total'] = sample['total'].cuda()
            output,mean,var = vae(sample['seed'])
            
            if total_anneal_steps > 0:
                anneal = min(anneal_cap, 1. * update_count / total_anneal_steps)
            else:
                anneal = anneal_cap

            loss = vae.loss_function(output,sample['total'],mean,var,anneal)
"""    
            l2_reg = torch.tensor(0.)
            for param in vae.parameters():
                l2_reg += torch.norm(param)
 """
            train_loss += loss 

            vae.zero_grad()
            loss.backward()
            optimizer.step()

            update_count += 1
            if (index+1) % 500 == 0:    #check the loss every 300*args.batchSize
                print('Epoch [%d/%d], Step[%d/%d], d_loss: %.4f'%(index_epoch, args.numEpoch , index+1,len(dataloader), train_loss.item()/500))
                train_loss = 0

            if(index+1) % 2000 == 0:   #check the performance every 3000*args.batchSize
                ndcg = []
                rpre = []
                qqq =0
                with torch.no_grad():
                    for _,sample_val in enumerate(vaildloader):
                        sample_val['seed'] = sample_val['seed'].cuda()
                        sample_val['total'] = sample_val['total'].cuda()
                        output_val,_,_ = vae(sample_val['seed'],training=False)
                        
                        output_val = output_val.cpu().detach().numpy()
                        ans_val = sample_val['total'].cpu().detach().numpy()
                        seed = sample_val['seed'].cpu().detach().numpy()
                        length_val = sample_val['length'].cpu().detach().numpy()

                        output_val[seed==1] = -np.inf
                        ans_val[seed==1] = 0


                        temp=(eval.NDCG_binary_at_k_batch(output_val,ans_val))
                        ndcg.append(temp)
                        temp = eval.Rpre_at_k_batch(output_val,ans_val,length_val)
                        rpre.extend(temp)
                        
                ndcg = np.concatenate(ndcg)
                rpre = np.concatenate(rpre)

                print("Test NDCG=%.5f (%.5f)" % (np.mean(ndcg), np.std(ndcg) / np.sqrt(ndcg.shape[0])))
                print("Test Rpre=%.5f (%.5f)" % (np.mean(rpre), np.std(rpre) / np.sqrt(rpre.shape[0])))

                if(best_ndcg < np.mean(ndcg) or best_rpre < np.mean(rpre)):
                    torch.save(vae.state_dict(), args.saveModel+'vae_params.pkl') # save only the parameters
                    best_ndcg = np.mean(ndcg)
                    best_rpre = np.mean(rpre)

if(__name__ == '__main__'):
    parser = argparse.ArgumentParser(description='Process some integers.')
    
    parser.add_argument('--numEpoch', type=int, default=10, help='input number of epoch')
    parser.add_argument('--batchSize', type=int, default=8, help='input batch size')
    parser.add_argument('--num_threads', type=int, default=8, help='# threads for loading data')
    parser.add_argument('--loadData', type=str, default="./data/pre_pytorch/", help='place to load data')
    parser.add_argument('--lr', type=float, default="0.001", help='learning for the optimizer')
    
    parser.add_argument('--loadModel', type=str, default="", help='place to load model')
    parser.add_argument('--saveModel', type=str, default="./checkpoint/", help='place to save model')
        

    args = parser.parse_args()
    main(args)
