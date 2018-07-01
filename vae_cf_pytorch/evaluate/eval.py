import numpy as np
import bottleneck as bn

def Rpre_at_k_batch(X_pred, heldout_batch,length):
    batch_users = X_pred.shape[0]
    X_pred_binary = np.zeros_like(X_pred, dtype=bool)

    X_true_binary = heldout_batch.astype('int')
    true_size = X_true_binary.sum(axis=1)

    for i in range(batch_users):
        if(true_size[i]==0):
            continue

        idx = bn.argpartition(-X_pred[i,:],length[i][0]-1)
        X_pred_binary[i, idx[:length[i][0]]] = True


    tmp = (np.logical_and(X_true_binary, X_pred_binary).sum(axis=1)).astype(np.float32)
    Rpre = tmp[:,np.newaxis] / (length+0.0000000000000000001)
    return Rpre

def NDCG_binary_at_k_batch(X_pred, heldout_batch,length, k=500):
    '''
    normalized discounted cumulative gain@k for binary relevance
    ASSUMPTIONS: all the 0's in heldout_data indicate 0 relevance
    '''
    batch_users = X_pred.shape[0]
    idx_topk_part = bn.argpartition(-X_pred, k-1, axis=1)
    topk_part = X_pred[np.arange(batch_users)[:, np.newaxis],idx_topk_part[:, :k]]

    idx_part = np.argsort(-topk_part, axis=1)
    
    # topk predicted score
    idx_topk = idx_topk_part[np.arange(batch_users)[:, np.newaxis], idx_part]
    # build the discount template
    tp = 1. / np.log2(np.arange(2, k + 2))

    ans = heldout_batch[np.arange(batch_users)[:,np.newaxis],idx_topk]
    
    click = []
    for data in ans:
        try:
            click.append(np.where(data==1)[0][0]//10+1)
        except IndexError:
            click.append(51)        
   
    DCG = (ans * tp).sum(axis=1)
     
    IDCG = np.array([(tp[:min(n[0], k)]).sum()  for n in length])

    return (DCG / (IDCG+0.00000000000000000000000001)),click

def output_ans(X_pred,pid,file_ans=0, k=500):
    '''
    normalized discounted cumulative gain@k for binary relevance
    ASSUMPTIONS: all the 0's in heldout_data indicate 0 relevance
    '''
    batch_users = X_pred.shape[0]
    
    idx_topk_part = bn.argpartition(-X_pred, k-1,axis=-1)

    topk_part = X_pred[np.arange(batch_users)[:, np.newaxis],idx_topk_part[:, :k]]
    idx_part = np.argsort(-topk_part, axis=1)

    # X_pred[np.arange(batch_users)[:, np.newaxis], idx_topk] is the sorted
    # topk predicted score
    idx_topk = idx_topk_part[np.arange(batch_users)[:, np.newaxis], idx_part]

    if(file_ans!=0):
        for i in range(idx_topk.shape[0]):
            file_ans.write('{0}'.format(pid[i]))
            for j in range(idx_topk.shape[1]):
                file_ans.write(' {0}'.format(idx_topk[i][j]))
            file_ans.write('\n')
