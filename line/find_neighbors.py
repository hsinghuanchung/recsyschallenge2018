import numpy as np
import pandas as pd

def find_neigh(emb, num_neigh, num_test):
	embb = np.array(emb)
	print ("dimensions: ", len(embb[0]))

	# KDTree (k neighbors, including self)
	from sklearn.neighbors import KDTree
	kdt = KDTree(embb, leaf_size=30, metric='euclidean')

	import random
	_index = random.sample(range(len(emb)), k=num_test)
	nearest = kdt.query(embb[_index], k=num_neigh, return_distance=False)
	print ("# of test songs: ", len(nearest))

	df = pd.DataFrame(nearest, index=0)
	return df

# Read in embedding
DIR = '.../'
file = 'bpr_1.txt'
emb = pd.read_csv(DIR+file, delimiter=' ', header=None, index_col=0)

# Find the 500 neighbors of the 1000 test nodes
df = find_neigh(emb, 500, 1000)
