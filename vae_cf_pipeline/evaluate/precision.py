import bottleneck as bn
import numpy as np

def Rpre_at_k_batch(X_pred, heldout_batch):
	batch_users = X_pred.shape[0]
	X_pred_binary = np.zeros_like(X_pred, dtype=bool)
    
	X_true_binary = (heldout_batch > 0).toarray()
	true_size = X_true_binary.sum(axis=1)
    
	for i in range(batch_users):
		if(true_size[i]==0):
			continue
        
		idx = bn.argpartition(-X_pred[i,:],true_size[i]-1)
		X_pred_binary[i, idx[:true_size[i]]] = True
	
	
	tmp = (np.logical_and(X_true_binary, X_pred_binary).sum(axis=1)).astype(np.float32)
	Rpre = tmp / (X_true_binary.sum(axis=1)+0.0000001)
	return Rpre

def NDCG_binary_at_k_batch(X_pred, heldout_batch, k=500):
	'''
	normalized discounted cumulative gain@k for binary relevance
	ASSUMPTIONS: all the 0's in heldout_data indicate 0 relevance
	'''
	batch_users = X_pred.shape[0]
	idx_topk_part = bn.argpartition(-X_pred, k, axis=1)
	topk_part = X_pred[np.arange(batch_users)[:, np.newaxis],idx_topk_part[:, :k]]

	idx_part = np.argsort(-topk_part, axis=1)
	# X_pred[np.arange(batch_users)[:, np.newaxis], idx_topk] is the sorted
	# topk predicted score
	idx_topk = idx_topk_part[np.arange(batch_users)[:, np.newaxis], idx_part]
	# build the discount template
	tp = 1. / np.log2(np.arange(2, k + 2))

	DCG = (heldout_batch[np.arange(batch_users)[:, np.newaxis],idx_topk].toarray() * tp).sum(axis=1)
	
	for n in heldout_batch.getnnz(axis=1):
		print(n)
	
	IDCG = np.array([(tp[:min(n, k)]).sum()  for n in heldout_batch.getnnz(axis=1)])
	
	return DCG / (IDCG)