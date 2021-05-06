import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as sch

cluster_id_verbose = "Cluster_Id"

def aggomerativeClustering(file_path, distance_mat, clusters):
    #dendrogram = sch.dendrogram(sch.linkage(distance_mat, method='ward'))
    model = AgglomerativeClustering(n_clusters = clusters, affinity='euclidean', linkage='ward')
    model.fit(distance_mat)
    labels = model.labels_
    return labels


def groupByClusters(labelled_weights):
    #import pdb;pdb.set_trace()
    #newFile = pd.read_csv(output_file, index_col = cluster_id_verbose)
    dfIds = labelled_weights.groupby(cluster_id_verbose).agg({'Id': list})
    dfWeights = labelled_weights.groupby(cluster_id_verbose)[labelled_weights.columns[:-1]].mean()
    del dfWeights['Id']
    dfFinal = dfIds.join(dfWeights)
    dfFinal.rename(columns = {'Id': 'GroupedIds'})
    return dfFinal

def redictributeWeights(file_name, clustered_weights):
    print("x")


