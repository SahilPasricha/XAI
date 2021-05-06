import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
import sys
import os
sys.path.append(os.getcwd())
from ReadCsv.read_csv import  CsvToArray
import pandas

def calc_reg(csv_file,pred_steps):
    #pred steps is number of steps to be predicted
    data_array = CsvToArray(csv_file)
    #count rows and columns
    rows = data_array.shape[0]
    cols = data_array.shape[1]
    #initialize an empty array to fill with predicted values
    predicted_array = np.empty((rows, (cols+pred_steps)))
    diabetes_X = data_array[0].reshape(-1,1)    
    diabetes_X_test = np.empty(pred_steps)
    #incrementing index array
    
    
    for i in range(0,pred_steps):
        diabetes_X_test[i]=cols+i   


    predicted_array[0] = np.around(np.concatenate((data_array[0].astype(int), diabetes_X_test.astype(int))),0)
    
    
    for i in range(1, (rows-1)):
        diabetes_y = np.around(data_array[i],4)

        # Create linear regression object
        regr = linear_model.LinearRegression()

        # Train the model using the training sets
        regr.fit(diabetes_X, diabetes_y)

        # Make predictions using the testing set
        diabetes_y_pred = regr.predict(diabetes_X_test.reshape(-1, 1))
        

        #Merge real values + predicted values and store to array that will be written to excel
        predicted_array[i] = np.around(np.concatenate((diabetes_y, diabetes_y_pred )),4)
        
       
    #retunr [predicted_cols, apart form row 0]
    for i in range(0,pred_steps):
        predicted_step = predicted_array[1:,(cols+i)]
        df = pandas.read_csv(csv_file)
        header = df.shape[1]
        df [header] = predicted_step
        df.to_csv(csv_file,index= False)
