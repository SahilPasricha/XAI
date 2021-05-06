import csv
from dtaidistance import dtw_visualisation as dtwvis
from dtaidistance import dtw
import numpy as np
from numpy import genfromtxt
from datetime import datetime




def readFile(file):
    #return(genfromtxt(file, dtype=float,delimiter=','))
    with open(file, 'r', encoding='utf-8-sig') as f:
        oh_all = np.genfromtxt(f, dtype=float, delimiter=',')

    return oh_all

def findDistance(values):
    #Random sampling amonsgt Rows
    #num_rows, num_cols = values.shape
    #rand_row_num = np.random.choice(num_rows, num_rows, replace = False)
    #values =  values[rand_row_num,:]
    now = datetime.now()
    print("Distace Cacluation Started at ",now.strftime("%H:%M:%S"))
    #import pdb;pdb.set_trace()
    ds = dtw.distance_matrix_fast(values)
    return ds
    #return ds,num_rows,rand_row_num

def cleanMatrix(dtw_matrix):

    rows = np.size(dtw_matrix,0)
    columns = np.size(dtw_matrix,1)
    for row in range(rows):
        for cells in range(columns):
            if(dtw_matrix[row,cells]== float("inf")):
                dtw_matrix[row,cells]=dtw_matrix[cells,row]
            if (row == cells):
                dtw_matrix[row,cells]=0
    return dtw_matrix


def ReadFileMain(filename):
    file = filename
    #Clean file of index row and column
    val = np.loadtxt(filename, delimiter=',', skiprows=1)
    val = np.delete(val, 0, axis = 1)
    #val = readFile(file)
    ds = findDistance(val)
    final_matrix = cleanMatrix(ds)
    return final_matrix


