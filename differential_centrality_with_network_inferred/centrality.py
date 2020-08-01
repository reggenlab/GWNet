import networkx as nx
import numpy as np
import pandas as pd
import sys, getopt
#from similarityMat import similarityMat

def get_top_vals(adj, top):
	adj = np.triu(adj)
	top = int(top/2)
	adj_return = np.zeros_like(adj)
	top_indices = np.unravel_index(np.argsort(adj, axis=None)[::-1][0:top], adj.shape)
	adj_return[top_indices] = adj[top_indices]
	adj_return = np.triu(adj_return) + np.transpose(np.triu(adj_return))
	return adj_return

def centrality(exp,top_val):
	#SimilarityMat = similarityMat()
	#if method == 'phi' or method == 'rho':
	#	exp[exp<0] = 0
	#adj = SimilarityMat.get_similarity_matrix(np.transpose(exp),method = method)
	adj = exp
	print(adj.shape)
	adj = get_top_vals(adj,top_val)
	G = nx.from_numpy_matrix(np.matrix(adj))
	#bt_c = nx.betweenness_centrality(G).values()
	pg_rank = nx.pagerank(G).values()
	degree_values = [v for k, v in nx.degree(G)]
	#cl_cn = nx.closeness_centrality(G).values()
	#L = nx.normalized_laplacian_matrix(G)
	#e = np.linalg.eigvals(L.A)
	#bt_c=list(bt_c)
	pg_rank=list(pg_rank)
	degree_values=list(degree_values)
	#cl_cn=list(cl_cn)
	#e=list(e)
	#return pg_rank,degree_values,bt_c,cl_cn,e
	return pg_rank,degree_values

def main(argv):
	result_path = 'diff_result.txt'
	exp_young = ''
	exp_old = ''
	top_edges = 10000000
	try:
		opts, args = getopt.getopt(argv,"hy:o:r:t:g:")
	except getopt.GetoptError:
		print("centrality.py -y <adjacency_young> -o <adjancency_old> -g <genes_list> -r <result_path> -t <top_edges>")
	for opt, arg in opts:
		if opt == '-h':
			print("centrality.py -y <adjacency_young> -o <adjacency_old> -g <genes_list> -r <result_path> -t <top_edges>")
			sys.exit()
		elif opt in ("-y"):
			exp_young = arg
		elif opt in ("-o"):
			exp_old = arg
		elif opt in ("-g"):
			genes_path = arg
		elif opt in ("-t"):
			top_edges = int(arg)

	if exp_young =='' or exp_old == '' or genes_path == '':
		print("centrality.py -y <adjacency_young> -o <adjacency_old>  -g <genes_list> -r <result_path> -t <top_edges>")
		print("Expression young, old and gene path required!!")
		sys.exit()
	else:
		exp_young = np.loadtxt(exp_young,delimiter=",")
		exp_old = np.loadtxt(exp_old,delimiter=",")
		genes = np.loadtxt(genes_path,delimiter=",",dtype='str')
		#genes = exp_young.index
		#pg_rank_1,degree_values_1,bt_c_1,cl_cn_1,e_1 = centrality(exp_young,top_val=top_edges)
		#pg_rank_2,degree_values_2,bt_c_2,cl_cn_2,e_2 = centrality(exp_old,top_val=top_edges)

		pg_rank_1,degree_values_1 = centrality(exp_young,top_val=top_edges)
		pg_rank_2,degree_values_2 = centrality(exp_old,top_val=top_edges)

		degree_difference = list()
		for i in range(len(degree_values_1)):
			degree_difference.append(degree_values_2[i]-degree_values_1[i])
		pg_difference = list()
		for i in range(len(pg_rank_2)):
			pg_difference.append(pg_rank_2[i]-pg_rank_1[i])

		final_diff= pd.concat([pd.Series(genes),pd.Series(degree_difference),pd.Series(pg_difference)],axis=1)
		final_diff.columns = ['genes','degree','pagerank']
		final_diff.to_csv(result_path,sep=",")

		#centrality_young= pd.concat([pd.Series(genes),pd.Series(degree_values_1),pd.Series(pg_rank_1),pd.Series(bt_c_1),pd.Series(cl_cn_1),pd.Series(e_1)],axis=1)
		#centrality_young.columns = ['genes','Degree','Pagerank','Betweenness','Closeness','Eigen values']
		#centrality_young.to_csv("./young_centrality.txt",sep=",")
		
		#centrality_old= pd.concat([pd.Series(genes),pd.Series(degree_values_2),pd.Series(pg_rank_2),pd.Series(bt_c_2),pd.Series(cl_cn_2),pd.Series(e_2)],axis=1)
		#centrality_old.columns = ['genes','Degree','Pagerank','Betweenness','Closeness','Eigen values']
		#centrality_old.to_csv("./old_centrality.txt",sep=",")

		centrality_young= pd.concat([pd.Series(genes),pd.Series(degree_values_1),pd.Series(pg_rank_1)],axis=1)
		centrality_young.columns = ['genes','Degree','Pagerank']
		centrality_young.to_csv("./young_centrality.txt",sep=",")
		
		centrality_old= pd.concat([pd.Series(genes),pd.Series(degree_values_2),pd.Series(pg_rank_2)],axis=1)
		centrality_old.columns = ['genes','Degree','Pagerank']
		centrality_old.to_csv("./old_centrality.txt",sep=",")

		
if __name__ == "__main__":
	main(sys.argv[1:])


