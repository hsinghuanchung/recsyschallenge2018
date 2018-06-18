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

        idx = bn.argpartition(-X_pred[i,:],true_size[i]-1)
        X_pred_binary[i, idx[:true_size[i]]] = True


    tmp = (np.logical_and(X_true_binary, X_pred_binary).sum(axis=1)).astype(np.float32)
    Rpre = tmp[:,np.newaxis] / (length-25+0.0000000000000000001)
    return Rpre

def NDCG_binary_at_k_batch(X_pred, heldout_batch, k=500):
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

    DCG = (heldout_batch[np.arange(batch_users)[:, np.newaxis],idx_topk] * tp).sum(axis=1)
        
    true_size = heldout_batch.astype('int').sum(axis=1)
    
    IDCG = np.array([(tp[:min(n, k)]).sum()  for n in true_size])

    return DCG / (IDCG+0.00000000000000000000000001)

def output_ans(X_pred,file_ans=0, k=500):
    '''
    normalized discounted cumulative gain@k for binary relevance
    ASSUMPTIONS: all the 0's in heldout_data indicate 0 relevance
    '''
    batch_users = X_pred.shape[0]
    print(X_pred.shape)
    idx_topk_part = bn.argpartition(-X_pred, k-1,axis=1)

    topk_part = X_pred[np.arange(batch_users)[:, np.newaxis],idx_topk_part[:, :k]]
    idx_part = np.argsort(-topk_part, axis=1)

    # X_pred[np.arange(batch_users)[:, np.newaxis], idx_topk] is the sorted
    # topk predicted score
    idx_topk = idx_topk_part[np.arange(batch_users)[:, np.newaxis], idx_part]

    if(file_ans!=0):
        for i in range(idx_topk.shape[0]):
            for j in range(idx_topk.shape[1]):
                file_ans.write('{0} '.format(idx_topk[i][j]))
            file_ans.write('\n')