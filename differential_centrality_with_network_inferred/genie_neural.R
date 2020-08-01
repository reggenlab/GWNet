args = commandArgs(trailingOnly = TRUE)
library("GENIE3")
require(methods)
setGeneric("GENIE3")
#setwd("/cellatlassearch_shreya/Network_Inference_neuron/")
expression = read.table(args[1], sep = "\t", row.names = 1 , header = TRUE)
set.seed(123) # For reproducibility of results
weightMat <- GENIE3(as.matrix(expression),nCores=30)
write.table(weightMat, file=args[2],sep = "\t", row.names=FALSE, col.names=FALSE)
