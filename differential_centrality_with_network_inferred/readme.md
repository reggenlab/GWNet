"# centrality.py" 
One needs to have python 3.0+ installed in their machine. Following are the dependencies of the code :
Numpy, Pandas, Pygsp, scipy, sklearn, networkx
R dependencies : minet,methods,GENIE3,propr

You have to download python code in your local machine/server.
For execution you have to pass filename of  young and old filtered expression csv as a input file. 
Expression csv file should not contain header and genes i.e. it consist of only data on which filtering is going to perform. 
Row represents samples and column represent genes in csv file. 

User gets :::
1. Differential degree and pagerank for old and young.
2. centrality (degree, pagerank) for both young and old in separate files by the default names 
"young_centrality.txt" and "old_centrality.txt"

USE THE FOLLOWING COMMANDS :
```bash
python3 centrality.py -y <filtered_young> -o <filtered_old> -g <genes_list> -c <Network_Inference_method> -r <result_path> -t <top_edges>		
e.g.
python3 centrality.py -y exp_young.txt -o exp_old.txt -g genes.txt -c pearson -r diff_output.txt -t 100000
```
'exp_young.txt' consist of filtered young expression data, 'exp_old.txt' consist of filtered old expression data.

Regarding each iput parameter :
-y <Filtered young expression Data> : Path to young filtered expression data 
-o <Filtered old expression Data> : Path to Old filtered expression data 
-g <List of genes in expression Data> : Path to Gene list
-r <result_path> : Path to save differential centrality result
-c <correlation_method> : Options : pearson, spearman, aracne, genie3, phi, rho
-t <top_edges> : Number of Top Edges to select to build network



