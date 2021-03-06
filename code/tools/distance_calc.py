
import pandas as pd
import pandas as pd
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
import numpy as np
from dtaidistance import dtw_visualisation as dtwvis
from dtaidistance import dtw
from fastdtw import fastdtw
import time

def dynamic_time_warping(input_file):
    print("***********inside DTw****************")
    data = pd.read_csv(input_file)
    #removing ids for dtw processing
    id_list = data['Id']
    del data['Id']
    values = np.array(data)
    start_time = time.clock()
    #import pdb;pdb.set_trace()
    #distance_mat = dtw.distance_matrix(values,parallel=True, compact=True)
    distance_mat = dtw.distance_matrix(values)
    print ("Distance Calc Took ",time.clock() - start_time, "seconds")
    distance_mat[distance_mat == np.inf] = 0
    distance_mat += distance_mat.T
    return id_list, distance_mat



