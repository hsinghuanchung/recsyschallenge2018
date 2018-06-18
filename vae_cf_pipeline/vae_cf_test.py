import os
import shutil
import sys
import numpy as np
from scipy import sparse
import matplotlib.pyplot as plt
import seaborn as sn
sn.set()
import pandas as pd
import tensorflow as tf
from tensorflow.contrib.layers import apply_regularization, l2_regularizer
import bottleneck as bn


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
### change `DATA_DIR` to the location where movielens-20m dataset sits
DATA_DIR = './data/'
pro_dir = os.path.join(DATA_DIR, 'pro_sg')
class MultiDAE(object):
    def __init__(self, p_dims, q_dims=None, lam=0.01, lr=1e-3, random_seed=None):
        self.p_dims = p_dims
        if q_dims is None:
            self.q_dims = p_dims[::-1]
        else:
            assert q_dims[0] == p_dims[-1], "Input and output dimension must equal each other for autoencoders."
            assert q_dims[-1] == p_dims[0], "Latent dimension for p- and q-network mismatches."
            self.q_dims = q_dims
        self.dims = self.q_dims + self.p_dims[1:]
        
        self.lam = lam
        self.lr = lr
        self.random_seed = random_seed

        self.construct_placeholders()

    def construct_placeholders(self):        
        self.input_ph = tf.placeholder(
            dtype=tf.float32, shape=[None, self.dims[0]])
        self.keep_prob_ph = tf.placeholder_with_default(1.0, shape=None)

    def build_graph(self):

        self.construct_weights()

        saver, logits = self.forward_pass()
        log_softmax_var = tf.nn.log_softmax(logits)

        # per-user average negative log-likelihood
        neg_ll = -tf.reduce_mean(tf.reduce_sum(log_softmax_var * self.input_ph, axis=1))
        # apply regularization to weights
        reg = l2_regularizer(self.lam)
        reg_var = apply_regularization(reg, self.weights)
        # tensorflow l2 regularization multiply 0.5 to the l2 norm
        # multiply 2 so that it is back in the same scale
        loss = neg_ll + 2 * reg_var
        
        train_op = tf.train.AdamOptimizer(self.lr).minimize(loss)

        # add summary statistics
        tf.summary.scalar('negative_multi_ll', neg_ll)
        tf.summary.scalar('loss', loss)
        merged = tf.summary.merge_all()
        return saver, logits, loss, train_op, merged

    def forward_pass(self):
        # construct forward graph        
        h = tf.nn.l2_normalize(self.input_ph, 1)
        h = tf.nn.dropout(h, self.keep_prob_ph)
        
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            h = tf.matmul(h, w) + b
            
            if i != len(self.weights) - 1:
                h = tf.nn.tanh(h)
        return tf.train.Saver(), h

    def construct_weights(self):

        self.weights = []
        self.biases = []
        
        # define weights
        for i, (d_in, d_out) in enumerate(zip(self.dims[:-1], self.dims[1:])):
            weight_key = "weight_{}to{}".format(i, i+1)
            bias_key = "bias_{}".format(i+1)
            
            self.weights.append(tf.get_variable(
                name=weight_key, shape=[d_in, d_out],
                initializer=tf.contrib.layers.xavier_initializer(
                    seed=self.random_seed)))
            
            self.biases.append(tf.get_variable(
                name=bias_key, shape=[d_out],
                initializer=tf.truncated_normal_initializer(
                    stddev=0.001, seed=self.random_seed)))
            
            # add summary stats
            tf.summary.histogram(weight_key, self.weights[-1])
            tf.summary.histogram(bias_key, self.biases[-1])

class MultiVAE(MultiDAE):

    def construct_placeholders(self):
        super(MultiVAE, self).construct_placeholders()

        # placeholders with default values when scoring
        self.is_training_ph = tf.placeholder_with_default(0., shape=None)
        self.anneal_ph = tf.placeholder_with_default(1., shape=None)
        
    def build_graph(self):
        self._construct_weights()

        saver, logits, KL = self.forward_pass()
        log_softmax_var = tf.nn.log_softmax(logits)

        neg_ll = -tf.reduce_mean(tf.reduce_sum(log_softmax_var * self.input_ph,
            axis=-1))
        # apply regularization to weights
        reg = l2_regularizer(self.lam)
        
        reg_var = apply_regularization(reg, self.weights_q + self.weights_p)
        # tensorflow l2 regularization multiply 0.5 to the l2 norm
        # multiply 2 so that it is back in the same scale
        neg_ELBO = neg_ll + self.anneal_ph * KL + 2 * reg_var
        
        train_op = tf.train.AdamOptimizer(self.lr).minimize(neg_ELBO)

        # add summary statistics
        tf.summary.scalar('negative_multi_ll', neg_ll)
        tf.summary.scalar('KL', KL)
        tf.summary.scalar('neg_ELBO_train', neg_ELBO)
        merged = tf.summary.merge_all()

        return saver, logits, neg_ELBO, train_op, merged
    
    def q_graph(self):
        mu_q, std_q, KL = None, None, None
        
        h = tf.nn.l2_normalize(self.input_ph, 1)
        h = tf.nn.dropout(h, self.keep_prob_ph)
        
        for i, (w, b) in enumerate(zip(self.weights_q, self.biases_q)):
            h = tf.matmul(h, w) + b
            
            if i != len(self.weights_q) - 1:
                h = tf.nn.tanh(h)
            else:
                mu_q = h[:, :self.q_dims[-1]]
                logvar_q = h[:, self.q_dims[-1]:]

                std_q = tf.exp(0.5 * logvar_q)
                KL = tf.reduce_mean(tf.reduce_sum(0.5 * (-logvar_q + tf.exp(logvar_q) + mu_q**2 - 1), axis=1))
        return mu_q, std_q, KL

    def p_graph(self, z):
        h = z
        
        for i, (w, b) in enumerate(zip(self.weights_p, self.biases_p)):
            h = tf.matmul(h, w) + b
            
            if i != len(self.weights_p) - 1:
                h = tf.nn.tanh(h)
        return h

    def forward_pass(self):
        # q-network
        mu_q, std_q, KL = self.q_graph()
        epsilon = tf.random_normal(tf.shape(std_q))

        sampled_z = mu_q + self.is_training_ph *\
            epsilon * std_q

        # p-network
        logits = self.p_graph(sampled_z)
        
        return tf.train.Saver(), logits, KL

    def _construct_weights(self):
        self.weights_q, self.biases_q = [], []
        
        for i, (d_in, d_out) in enumerate(zip(self.q_dims[:-1], self.q_dims[1:])):
            if i == len(self.q_dims[:-1]) - 1:
                # we need two sets of parameters for mean and variance,
                # respectively
                d_out *= 2
            weight_key = "weight_q_{}to{}".format(i, i+1)
            bias_key = "bias_q_{}".format(i+1)
            
            self.weights_q.append(tf.get_variable(
                name=weight_key, shape=[d_in, d_out],
                initializer=tf.contrib.layers.xavier_initializer(
                    seed=self.random_seed)))
            
            self.biases_q.append(tf.get_variable(
                name=bias_key, shape=[d_out],
                initializer=tf.truncated_normal_initializer(
                    stddev=0.001, seed=self.random_seed)))
            
            # add summary stats
            tf.summary.histogram(weight_key, self.weights_q[-1])
            tf.summary.histogram(bias_key, self.biases_q[-1])
            
        self.weights_p, self.biases_p = [], []

        for i, (d_in, d_out) in enumerate(zip(self.p_dims[:-1], self.p_dims[1:])):
            weight_key = "weight_p_{}to{}".format(i, i+1)
            bias_key = "bias_p_{}".format(i+1)
            self.weights_p.append(tf.get_variable(
                name=weight_key, shape=[d_in, d_out],
                initializer=tf.contrib.layers.xavier_initializer(
                    seed=self.random_seed)))
            
            self.biases_p.append(tf.get_variable(
                name=bias_key, shape=[d_out],
                initializer=tf.truncated_normal_initializer(
                    stddev=0.001, seed=self.random_seed)))
            
            # add summary stats
            tf.summary.histogram(weight_key, self.weights_p[-1])
            tf.summary.histogram(bias_key, self.biases_p[-1])

unique_sid = list()
with open(os.path.join(pro_dir, 'unique_sid.txt'), 'r') as f:
    for line in f:
        unique_sid.append(line.strip())

n_items = 167985

def load_train_data(data_file):
	tp = open(data_file,'r')

	indptr_list = [0]
	indices_list = []
    
	for line in tp:
		line = line.strip().split()
		indptr_list.append(indptr_list[-1]+len(line)-1)
        
		for data in line[1:]:
			indices_list.append(int(data))
    
	indptr = np.array(indptr_list)
	indices = np.array(indices_list)
	data = np.ones_like(indices)

    
	n_users = len(indptr)-1
	
	data = sparse.csr_matrix(( data,indices,indptr), dtype='float64',shape=(n_users, n_items))
	return data


def load_tr_te_data(data_file_tr, data_file_te):
	tp_tr = open(data_file_tr,'r')
	tp_te = open(data_file_te,'r')
    
	indptr_tr_list,indptr_te_list = [0],[0]
	indices_tr_list,indices_te_list = [],[]

	for line_tr,line_te in zip(tp_tr,tp_te):
		line_tr = line_tr.strip().split()
		line_te = line_te.strip().split()
        
		if(len(line_tr)==1 or len(line_te)==1):
			continue

		indptr_tr_list.append(indptr_tr_list[-1]+len(line_tr)-1)
		indptr_te_list.append(indptr_te_list[-1]+len(line_te)-1)
        
		for data in line_tr[1:]:
			indices_tr_list.append(int(data))
        
		for data in line_te[1:]:
			indices_te_list.append(int(data))
    
	n_users = len(indptr_tr_list)-1

	indptr = np.array(indptr_tr_list)
	indices = np.array(indices_tr_list)
	data = np.ones_like(indices)
	data_tr = sparse.csr_matrix(( data,indices,indptr), dtype='float64',shape=(n_users, n_items))
    
	indptr = np.array(indptr_te_list)
	indices = np.array(indices_te_list)
	data = np.ones_like(indices)
	data_te = sparse.csr_matrix(( data,indices,indptr), dtype='float64',shape=(n_users, n_items))
    
	return data_tr, data_te


# training batch size
batch_size = 500
# the total number of gradient updates for annealing
total_anneal_steps = 200000
# largest annealing parameter
anneal_cap = 0.2

def NDCG_binary_at_k_batch(X_pred, heldout_batch, k=500,file_ans=0):
    '''
    normalized discounted cumulative gain@k for binary relevance
    ASSUMPTIONS: all the 0's in heldout_data indicate 0 relevance
    '''
    batch_users = X_pred.shape[0]
    idx_topk_part = bn.argpartition(-X_pred, k-1, axis=1)

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


    # build the discount template
    tp = 1. / np.log2(np.arange(2, k + 2))

    DCG = (heldout_batch[np.arange(batch_users)[:, np.newaxis],idx_topk].toarray() * tp).sum(axis=1)
    IDCG = np.array([(tp[:min(n, k)]).sum()for n in heldout_batch.getnnz(axis=1)])

    return DCG / IDCG

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
	return tmp / (X_true_binary.sum(axis=1)+0.0000001)

p_dims = [300, 800, n_items]

test_data_tr,test_data_te = load_tr_te_data(os.path.join(pro_dir, 'test_tr.csv'),os.path.join(pro_dir, 'test_te.csv'))

N_test = test_data_tr.shape[0]
idxlist_test = range(N_test)
batch_size_test = 2000

tf.reset_default_graph()
vae = MultiVAE(p_dims, lam=0.0)
saver, logits_var, _, _, _ = vae.build_graph()    

arch_str = "I-%s-I" % ('-'.join([str(d) for d in vae.dims[1:-1]]))
chkpt_dir = './VAE_anneal{}K_cap{:1.1E}/{}'.format(total_anneal_steps/1000, anneal_cap, arch_str)
print("chkpt directory: %s" % chkpt_dir)

f_ans = open("ans.out",'w')

n500_list, r_list = [], []
with tf.Session() as sess:
    saver.restore(sess, '{}/model'.format(chkpt_dir))

    for bnum, st_idx in enumerate(range(0, N_test, batch_size_test)):
        print(bnum)
        end_idx = min(st_idx + batch_size_test, N_test)
        X = test_data_tr[idxlist_test[st_idx:end_idx]]

        if sparse.isspmatrix(X):
            X = X.toarray()
        X = X.astype('float32')

        pred_val = sess.run(logits_var, feed_dict={vae.input_ph: X})
        # exclude examples from training and validation (if any)
        pred_val[X.nonzero()] = -np.inf

        n500_list.append(NDCG_binary_at_k_batch(pred_val, test_data_te[idxlist_test[st_idx:end_idx]], k=500,file_ans=f_ans))
        r_list.append(Rpre_at_k_batch(pred_val, test_data_te[idxlist_test[st_idx:end_idx]]))
        
n500_list = np.concatenate(n500_list)
r_list = np.concatenate(r_list)

print("Test NDCG@500=%.5f (%.5f)" % (np.mean(n500_list), np.std(n500_list) / np.sqrt(len(n500_list))))
print("Test Rpre=%.5f (%.5f)" % (np.mean(r_list), np.std(r_list) / np.sqrt(len(r_list))))
