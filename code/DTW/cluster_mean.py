import pandas as pd
import numpy 
from sklearn.linear_model import LinearRegression

pred_steps = 10
cluster_count = 0


def mean(file_name):
    clusters = pd.read_csv(file_name)
    header = numpy.append( clusters.columns.values[:-1],numpy.arange(last_elem+1,last_elem+1+pred_steps))
    last_elem = int(header[-1])
    
    clusters.drop('Id', axis = 'columns')
    cluster_count = clusters['cluster_id'].nunique()
    cluster_mean = clusters.groupby(['cluster_id']).mean()
    #perform regrssion on every cluster 
    reg_clusters = regression(cluster_mean)
    raw = pd.read_csv(file_name).to_numpy()
    
    header = clusters.columns.values[:-1]
    #reg_raw = numpy.empty((raw.shape[0], raw.shape[1]+pred_steps-1 ), dtype=float, order='C')
    reg_raw = numpy.empty((raw.shape[0], raw.shape[1]+pred_steps-1), dtype=float, order='C')
    for i in range(raw.shape[0]):
       # cluster_id = int(raw[i][-1])  
        cluster_id = int(raw[i][-1])
        # conccat --> raw data (leave clsuter id) , Last pre_setps vars of predicted array
        #reg_raw[i] = numpy.append(raw[i][:-1], reg_clusters[cluster_id][-pred_steps:])
        reg_raw[i] = numpy.append(raw[i][:-1], reg_clusters[cluster_id][-pred_steps:]).round(decimals=4)
    
    #convert 1st col to int
    
    #save this
    import pdb;pdb.set_trace()
    #numpy.savetxt(file_name, reg_raw, delimiter=",", fmt='%f')
    df = pd.DataFrame(data=reg_raw, columns=header)
    df.to_csv(file_name, index=False)
    
def regression(cluster_mean):
    cluster_count = cluster_mean.shape[0]
    reg_data = numpy.empty((cluster_count, cluster_mean.shape[1]+pred_steps ), dtype=float, order='C')
    data = cluster_mean.to_numpy()
    for i in range(cluster_count):
        dep = data [i]
        indep = numpy.arange(0,dep.shape[0])
        reg = LinearRegression().fit(indep.reshape(-1,1), dep.reshape(-1,1))
        pred_indep =  numpy.arange( indep[-1], indep[-1]+10)
        predicted= reg.predict(pred_indep.reshape(-1,1))
        reg_data[i]= numpy.append(dep,predicted)
        
    return reg_data
    