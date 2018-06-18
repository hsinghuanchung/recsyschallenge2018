import os
import shutil
import sys
import numpy as np
from scipy import sparse
import matplotlib.pyplot as plt
import seaborn as sn
sn.set()
import pandas as pd
import bottleneck as bn
import tensorflow as tf

from model.VAE import MultiVAE
from evaluate.precision import NDCG_binary_at_k_batch,Rpre_at_k_batch

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
n_items = 167985


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

def numerize(tp,file_name):
	f = open(file_name,'w')
    
	for i,line in enumerate(tp):
		f.write("{0} ".format(i))
		f.write(" ".join([ str(_) for _ in line]))
		f.write("\n")

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

def data_prepare():
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

	vad_plays = [ [ data for data in raw_data[_] if(data in unique_sid) ] for _ in vd_users  ]

	vad_plays_tr, vad_plays_te = split_train_test_proportion(vad_plays)
	test_plays = [ [ data for data in raw_data[_] if(data in unique_sid) ] for _ in te_users  ]
	test_plays_tr, test_plays_te = split_train_test_proportion(test_plays)
    
	numerize(train_plays,os.path.join(pro_dir, 'train.csv'))
	numerize(vad_plays_tr,os.path.join(pro_dir, 'validation_tr.csv'))
	numerize(vad_plays_te,os.path.join(pro_dir, 'validation_te.csv'))
	numerize(test_plays_tr,os.path.join(pro_dir, 'test_tr.csv'))
	numerize(test_plays_te,os.path.join(pro_dir, 'test_te.csv'))
 
def read_data():
	unique_sid = list()
	DATA_DIR = './data/'
	pro_dir = os.path.join(DATA_DIR, 'pro_sg')
	with open(os.path.join(pro_dir, 'unique_sid.txt'), 'r') as f:
		for line in f:
			unique_sid.append(line.strip())

	print('n_item',n_items)

	train_data = load_train_data(os.path.join(pro_dir, 'train.csv'))
	
	vad_data_tr, vad_data_te = load_tr_te_data(os.path.join(pro_dir, 'validation_tr.csv'),os.path.join(pro_dir, 'validation_te.csv'))
	
	return train_data,vad_data_tr,vad_data_te

def training(train_data,vad_data_tr,vad_data_te,loading=False):
	N = train_data.shape[0]
	idxlist = range(N)

	# training batch size
	batch_size = 500
	batches_per_epoch = int(np.ceil(float(N) / batch_size))

	N_vad = vad_data_tr.shape[0]
	idxlist_vad = range(N_vad)

	# validation batch size (since the entire validation set might not fit into GPU memory)
	batch_size_vad = 200

	# the total number of gradient updates for annealing
	# largest annealing parameter
	total_anneal_steps = 200000
	anneal_cap = 0.2

	n_items = 167985
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

	log_dir = './VAE_anneal{}K_cap{:1.1E}/{}'.format(total_anneal_steps/1000, anneal_cap, arch_str)

	print("log directory: %s" % log_dir)
	summary_writer = tf.summary.FileWriter(log_dir, graph=tf.get_default_graph())

	chkpt_dir = './VAE_anneal{}K_cap{:1.1E}/{}'.format(total_anneal_steps/1000, anneal_cap, arch_str)

	if not os.path.isdir(chkpt_dir):
		os.makedirs(chkpt_dir) 
		
	print("chkpt directory: %s" % chkpt_dir)

	n_epochs = 10

	ndcgs_vad = []
	rpres_vad = []
	with tf.Session() as sess:
		if(loading):
			saver.restore(sess, '{}/model'.format(chkpt_dir))
			graph = tf.get_default_graph()
		else:
			if os.path.exists(log_dir):
				shutil.rmtree(log_dir)
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

def testing():
	test_data_tr, test_data_te = load_tr_te_data(os.path.join(pro_dir, 'test_tr.csv'),os.path.join(pro_dir, 'test_te.csv'))
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
	print("Test NDCG=%.5f (%.5f)" % (np.mean(n500_list), np.std(n500_list) / np.sqrt(len(n500_list))))
	print("Test Rpre=%.5f (%.5f)" % (np.mean(r_list), np.std(r_list) / np.sqrt(len(r_list))))


if(__name__ == '__main__'):
	try:
		if(sys.argv[1] == 'train'):
			train_flag = True
		elif(sys.argv[1] == 'test'):
			train_flag = False
	
		if(sys.argv[2] == 'load'):
			load_flag = False
		elif(sys.argv[2] == 'not'):
			load_flag = True
	
	except IndexError:
		print('please input train or test / load or not')
		sys.exit(0)

	if(train_flag and load_flag):
		data_prepare()
	elif(train_flag):
		train_data,vad_data_tr,vad_data_te = read_data()
		
		if(not load_flag):
			print('you choose to train with load')
			training(train_data,vad_data_tr,vad_data_te,True)
		else:
			print('you choose to start a new training')
			training(train_data,vad_data_tr,vad_data_te,False)
	else:
		testing()
