import numpy as np
import matplotlib.pyplot as plt
import pygsp, csv,random, os, sys, getopt
from pygsp import graphs, filters
from scipy.stats import spearmanr
from scipy.integrate import simps
from pygsp import graphs, filters
from sklearn.manifold import TSNE
from pygsp import utils
from sklearn.metrics import mutual_info_score
import pandas as pd
from sklearn.neighbors import kneighbors_graph

class similarityMat:
	def __init__(self):
		pass

	def calc_MI(self,x, y, bins):
		c_xy = np.histogram2d(x, y, bins)[0]
		mi = mutual_info_score(None, None, contingency=c_xy)
		return mi

	def get_similarity_matrix(self,exp, method, subsample_size=5, bins = 10):
		if method == 'pearson':
			adj = np.corrcoef(np.transpose(exp))
		elif method == 'spearman':
			adj, p = spearmanr(exp)
		elif method == 'phi':
			np.savetxt('exp.txt', exp, delimiter='\t')
			os.system("Rscript phi.R exp.txt correlated_mat.txt")
			adj = np.genfromtxt('correlated_mat.txt', delimiter='\t')
		elif method == 'rho':
			np.savetxt('exp.txt', exp, delimiter='\t')
			os.system("Rscript rho.R exp.txt correlated_mat.txt")
			adj = np.genfromtxt('correlated_mat.txt', delimiter='\t')
		elif method == 'MI':
			adj = np.zeros((exp.shape[1], exp.shape[1]))
			for i in range(exp.shape[1]-1):
				print(i)
				for j in range(i, exp.shape[1]):
					samples = np.random.choice(exp.shape[0], subsample_size, replace=False)
					adj[i, j] = ee.mi(ee.vectorize(exp[samples, i]), ee.vectorize(exp[samples, j]))
		elif method == 'partial_corr':
			exp_sub = exp[np.random.choice(exp.shape[0], subsample_size, replace=False), :]
			print(exp_sub.shape)
			adj = pc.partial_corr(exp_sub)
		elif method == 'genie3':
			exp1 = np.transpose(exp)
			print("exp1 shape:::",exp1.shape)
			exp1 = np.insert(exp1 , 0 , range(1,exp1.shape[1]+1), axis = 0)
			exp1 = np.insert(exp1 , 0 , range(0,exp1.shape[0]), axis = 1)
			np.savetxt('exp.txt', exp1, delimiter='\t')
			os.system("Rscript genie_neural.R exp.txt correlated_mat.txt")
			adj = np.genfromtxt('correlated_mat.txt', delimiter='\t',filling_values=0)
			print("adj shape :::",adj.shape)
		elif method == 'aracne':
			adj_mi = np.zeros((exp.shape[1], exp.shape[1]))
			for ix in np.arange(exp.shape[1]):
				for jx in np.arange(ix + 1, exp.shape[1]):
					adj_mi[ix, jx] = self.calc_MI(exp[:, ix], exp[:, jx], bins)
			np.savetxt('exp.txt', adj_mi, delimiter='\t')
			os.system("Rscript aracne.R exp.txt correlated_mat.txt")
			adj = np.genfromtxt('correlated_mat.txt', delimiter='\t')
		else:
			print("Unkown input!")
		adj = np.nan_to_num(adj)
		adj = np.abs(adj)                                   # no negative weights
		adj = adj - np.diag(np.diag(adj))                   # no self loops
		adj = np.triu(adj) + np.transpose(np.triu(adj))     # no directions
		return adj
