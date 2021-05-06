import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as sch
from ReadData import ReadFileMain
#from addcolumns import add_column_in_csv
from cluster_mean import mean
file_dir = "/home/pasricha/venv/log/dtw_exp/weights/"
file_name = "Rweights00.csv"

data_file = file_dir +file_name
output_file =  file_dir + "clustered_" + file_name
print(output_file)

# returns the dtw distance
import pdb;pdb.set_trace()
X = ReadFileMain(data_file)
distance_file = file_dir + "distance_" + file_name
#np.savetxt(distance_file, X, delimiter =',')
dendrogram = sch.dendrogram(sch.linkage(X, method='ward'))
model = AgglomerativeClustering(n_clusters= 200, affinity='euclidean', linkage='ward')
model.fit(X)
labels = model.labels_


add_column_in_csv(data_file, output_file, lambda row, line_num: row.append(labels[line_num - 1]))

newFile = pd.read_csv(output_file, index_col = 'cluster_id')
dfIds = newFile.groupby("cluster_id").agg({'Id': list})
dfWeights = newFile.groupby("cluster_id")[newFile.columns[:-1]].mean()
df.to_csv(file_dir + "test.csv")

del dfWeights['Id']
dfFinal = dfIds.join(dfWeights)
# mean(output_file)