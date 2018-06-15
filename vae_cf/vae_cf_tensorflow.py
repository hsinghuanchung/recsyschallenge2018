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

raw_file = open(os.path.join(DATA_DIR, 'training_data25'), 'r')
raw_data = []
for line in raw_file:
	raw_data.append([ int(_) for _ in line.strip().split()])

unique_uid = np.array(list( range(len(raw_data))))

np.random.seed(98765)
idx_perm = np.random.permutation(unique_uid.size)
unique_uid = unique_uid[idx_perm]

# create train/validation/test users
n_users = unique_uid.size
n_heldout_users = 10000

tr_users = unique_uid[:(n_users - n_heldout_users * 2)]
vd_users = unique_uid[(n_users - n_heldout_users * 2): (n_users - n_heldout_users)]
te_users = unique_uid[(n_users - n_heldout_users):]

train_plays = [ raw_data[_] for _ in tr_users ]

unique_sid = set()
for line in train_plays:
	for data in line:
		unique_sid.add(data)

pro_dir = os.path.join(DATA_DIR, 'pro_sg')
if not os.path.exists(pro_dir):
	os.makedirs(pro_dir)

with open(os.path.join(pro_dir, 'unique_sid.txt'), 'w') as f:
	for sid in unique_sid:
		f.write('%s\n' % sid)

def split_train_test_proportion(data, test_prop=25):
	tr_list, te_list = list(), list()

	np.random.seed(98765)

	for i, group in enumerate(data):
		n_items_u = len(group)
		
		if n_items_u > 25:
			idx = np.zeros(n_items_u, dtype='bool')
			idx[np.random.choice(n_items_u, size=test_prop, replace=False).astype('int64')] = True
			
			tr_arr = []
			te_arr = []
			
			for index,ans_idx in enumerate(idx):
				if(ans_idx):
					tr_arr.append(group[index])
				else:
					te_arr.append(group[index])
                    
			tr_list.append(tr_arr)
			te_list.append(te_arr)
		else:
			print('error\n')

		if i % 1000 == 0:
			print("%d users sampled" % i)
			sys.stdout.flush()
    
	return tr_list, te_list

vad_plays = [ [ data for data in raw_data[_] if(data in unique_sid) ] for _ in vd_users  ]

vad_plays_tr, vad_plays_te = split_train_test_proportion(vad_plays)
test_plays = [ [ data for data in raw_data[_] if(data in unique_sid) ] for _ in te_users  ]
test_plays_tr, test_plays_te = split_train_test_proportion(test_plays)

def numerize(tp,file_name):
	f = open(file_name,'w')
    
	for i,line in enumerate(tp):
		f.write("{0} ".format(i))
		f.write(" ".join([ str(_) for _ in line]))
		f.write("\n")
    
numerize(train_plays,os.path.join(pro_dir, 'train.csv'))
numerize(vad_plays_tr,os.path.join(pro_dir, 'validation_tr.csv'))
numerize(vad_plays_te,os.path.join(pro_dir, 'validation_te.csv'))
numerize(test_plays_tr,os.path.join(pro_dir, 'test_tr.csv'))
numerize(test_plays_te,os.path.join(pro_dir, 'test_te.csv'))

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
		self.input_ph = tf.placeholder(dtype=tf.float32, shape=[None, self.dims[0]])
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
            
			self.weights.append(tf.get_variable(ame=weight_key, shape=[d_in, d_out],initializer=tf.contrib.layers.xavier_initializer(seed=self.random_seed)))
            
			self.biases.append(tf.get_variable(name=bias_key, shape=[d_out],initializer=tf.truncated_normal_initializer(stddev=0.001, seed=self.random_seed)))
            
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
		neg_ll = -tf.reduce_mean(tf.reduce_sum(log_softmax_var * self.input_ph,axis=-1))

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

		sampled_z = mu_q + self.is_training_ph * epsilon * std_q

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
			
			self.weights_q.append(tf.get_variable(name=weight_key, shape=[d_in, d_out],initializer=tf.contrib.layers.xavier_initializer(seed=self.random_seed)))
            
			self.biases_q.append(tf.get_variable(name=bias_key, shape=[d_out],initializer=tf.truncated_normal_initializer(stddev=0.001, seed=self.random_seed)))
            
            # add summary stats
			tf.summary.histogram(weight_key, self.weights_q[-1])
			tf.summary.histogram(bias_key, self.biases_q[-1])
            
		self.weights_p, self.biases_p = [], []
		
		for i, (d_in, d_out) in enumerate(zip(self.p_dims[:-1], self.p_dims[1:])):
			weight_key = "weight_p_{}to{}".format(i, i+1)
			bias_key = "bias_p_{}".format(i+1)
			self.weights_p.append(tf.get_variable(name=weight_key, shape=[d_in, d_out],initializer=tf.contrib.layers.xavier_initializer(seed=self.random_seed)))
            
			self.biases_p.append(tf.get_variable(name=bias_key, shape=[d_out],initializer=tf.truncated_normal_initializer(stddev=0.001, seed=self.random_seed)))
            
			# add summary stats
			tf.summary.histogram(weight_key, self.weights_p[-1])
			tf.summary.histogram(bias_key, self.biases_p[-1])

unique_sid = list()
with open(os.path.join(pro_dir, 'unique_sid.txt'), 'r') as f:
	for line in f:
		unique_sid.append(line.strip())

n_items = 167985
print('n_item',n_items)
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


train_data = load_train_data(os.path.join(pro_dir, 'train.csv'))
def load_tr_te_data(data_file_tr, data_file_te):
    
	tp_tr = open(data_file_tr,'r')
	tp_te = open(data_file_te,'r')
    
	indptr_tr_list = [0]
	indices_tr_list = []
    
	indptr_te_list = [0]
	indices_te_list = []

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
    
vad_data_tr, vad_data_te = load_tr_te_data(os.path.join(pro_dir, 'validation_tr.csv'),
                                           os.path.join(pro_dir, 'validation_te.csv'))

N = train_data.shape[0]
idxlist = range(N)

# training batch size
batch_size = 500
batches_per_epoch = int(np.ceil(float(N) / batch_size))

N_vad = vad_data_tr.shape[0]
idxlist_vad = range(N_vad)

# validation batch size (since the entire validation set might not fit into GPU memory)
batch_size_vad = 2000

# the total number of gradient updates for annealing
total_anneal_steps = 200000
# largest annealing parameter
anneal_cap = 0.2

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
	tp = np.hstack((np.array([1]),1. / np.log2(np.arange(2, k + 1))))

	DCG = (heldout_batch[np.arange(batch_users)[:, np.newaxis],idx_topk].toarray() * tp).sum(axis=1)
	
	for n in heldout_batch.getnnz(axis=1):
		print(n)
	
	IDCG = np.array([(tp[:min(n, k)]).sum()  for n in heldout_batch.getnnz(axis=1)])
	
	return DCG / (IDCG)

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

p_dims = [300, 800, n_items]

tf.reset_default_graph()
vae = MultiVAE(p_dims, lam=0.0, random_seed=98765)

saver, logits_var, loss_var, train_op_var, merged_var = vae.build_graph()

ndcg_var = tf.Variable(0.0)
ndcg_dist_var = tf.placeholder(dtype=tf.float64, shape=None)
ndcg_summary = tf.summary.scalar('ndcg_at_k_validation', ndcg_var)
ndcg_dist_summary = tf.summary.histogram('ndcg_at_k_hist_validation', ndcg_dist_var)

rpre_var = tf.Variable(0.0)
rpre_dist_var = tf.placeholder(dtype=tf.float64, shape=None)
rpre_summary = tf.summary.scalar('rpre_at_k_validation', rpre_var)
rpre_dist_summary = tf.summary.histogram('rpre_at_k_hist_validation', rpre_dist_var)

merged_valid = tf.summary.merge([ndcg_summary, ndcg_dist_summary,rpre_summary, rpre_dist_summary])

arch_str = "I-%s-I" % ('-'.join([str(d) for d in vae.dims[1:-1]]))

log_dir = './VAE_anneal{}K_cap{:1.1E}/{}'.format(
    total_anneal_steps/1000, anneal_cap, arch_str)

if os.path.exists(log_dir):
    shutil.rmtree(log_dir)

print("log directory: %s" % log_dir)
summary_writer = tf.summary.FileWriter(log_dir, graph=tf.get_default_graph())

chkpt_dir = './VAE_anneal{}K_cap{:1.1E}/{}'.format(
    total_anneal_steps/1000, anneal_cap, arch_str)

if not os.path.isdir(chkpt_dir):
    os.makedirs(chkpt_dir) 
    
print("chkpt directory: %s" % chkpt_dir)

n_epochs = 10

ndcgs_vad = []
rpres_vad = []
with tf.Session() as sess:

    init = tf.global_variables_initializer()
    sess.run(init)

    best_ndcg = -np.inf

    update_count = 0.0
    
    for epoch in range(n_epochs):
        np.random.shuffle(list(idxlist))
        # train for one epoch
        for bnum, st_idx in enumerate(range(0, N, batch_size)):
            end_idx = min(st_idx + batch_size, N)
            X = train_data[idxlist[st_idx:end_idx]]
            
            if sparse.isspmatrix(X):
                X = X.toarray()
            X = X.astype('float32')           
            
            if total_anneal_steps > 0:
                anneal = min(anneal_cap, 1. * update_count / total_anneal_steps)
            else:
                anneal = anneal_cap
            
            feed_dict = {vae.input_ph: X, 
                         vae.keep_prob_ph: 0.5, 
                         vae.anneal_ph: anneal,
                         vae.is_training_ph: 1}        
            sess.run(train_op_var, feed_dict=feed_dict)

            if bnum % 100 == 0:
                summary_train = sess.run(merged_var, feed_dict=feed_dict)
                summary_writer.add_summary(summary_train,global_step=epoch * batches_per_epoch + bnum) 
                print(epoch,bnum)
            update_count += 1
        
        # compute validation NDCG
        ndcg_dist = []
        rpre_dist = []
        for bnum, st_idx in enumerate(range(0, N_vad, batch_size_vad)):
            end_idx = min(st_idx + batch_size_vad, N_vad)
            X = vad_data_tr[idxlist_vad[st_idx:end_idx]]

            if sparse.isspmatrix(X):
                X = X.toarray()
            X = X.astype('float32')
        
            pred_val = sess.run(logits_var, feed_dict={vae.input_ph: X} )
            # exclude examples from training and validation (if any)
            pred_val[X.nonzero()] = -np.inf
            ndcg_dist.append(NDCG_binary_at_k_batch(pred_val, vad_data_te[idxlist_vad[st_idx:end_idx]]))
            rpre_dist.append(Rpre_at_k_batch(pred_val, vad_data_te[idxlist_vad[st_idx:end_idx]]))
        
        ndcg_dist = np.concatenate(ndcg_dist)
        ndcg_ = ndcg_dist.mean()
        ndcgs_vad.append(ndcg_)
        
        rpre_dist = np.concatenate(rpre_dist)
        rpre_ = rpre_dist.mean()
        rpres_vad.append(rpre_)
        
        merged_valid_val = sess.run(merged_valid, feed_dict={ndcg_var: ndcg_, ndcg_dist_var: ndcg_dist,rpre_var: rpre_, rpre_dist_var: rpre_dist})

        summary_writer.add_summary(merged_valid_val, epoch)

        # update the best model (if necessary)
        if ndcg_ > best_ndcg:
            saver.save(sess, '{}/model'.format(chkpt_dir))
            best_ndcg = ndcg_

plt.figure(figsize=(12, 3))
plt.plot(ndcgs_vad)
plt.ylabel("Validation NDCG@500")
plt.xlabel("Epochs")
pass

test_data_tr, test_data_te = load_tr_te_data(
    os.path.join(pro_dir, 'test_tr.csv'),
    os.path.join(pro_dir, 'test_te.csv'))

N_test = test_data_tr.shape[0]
idxlist_test = range(N_test)

batch_size_test = 2000


tf.reset_default_graph()
vae = MultiVAE(p_dims, lam=0.0)
saver, logits_var, _, _, _ = vae.build_graph()    

chkpt_dir = './VAE_anneal{}K_cap{:1.1E}/{}'.format(total_anneal_steps/1000, anneal_cap, arch_str)
print("chkpt directory: %s" % chkpt_dir)

n500_list, r_list = [], []

with tf.Session() as sess:
    saver.restore(sess, '{}/model'.format(chkpt_dir))

    for bnum, st_idx in enumerate(range(0, N_test, batch_size_test)):
        end_idx = min(st_idx + batch_size_test, N_test)
        X = test_data_tr[idxlist_test[st_idx:end_idx]]

        if sparse.isspmatrix(X):
            X = X.toarray()
        X = X.astype('float32')

        pred_val = sess.run(logits_var, feed_dict={vae.input_ph: X})
        # exclude examples from training and validation (if any)
        pred_val[X.nonzero()] = -np.inf
        n500_list.append(NDCG_binary_at_k_batch(pred_val, test_data_te[idxlist_test[st_idx:end_idx]], k=500))
        r_list.append(Rpre_at_k_batch(pred_val, test_data_te[idxlist_test[st_idx:end_idx]]))
        
n500_list = np.concatenate(n500_list)
r_list = np.concatenate(r_list)

print("Test NDCG@100=%.5f (%.5f)" % (np.mean(n500_list), np.std(n500_list) / np.sqrt(len(n500_list))))
print("Test Rpre@20=%.5f (%.5f)" % (np.mean(r_list), np.std(r_list) / np.sqrt(len(r_list))))





