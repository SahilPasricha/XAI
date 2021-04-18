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
    predicted_array = np.empty((rows, (cols+pred_steps-1)))
    diabetes_X = data_array[0].reshape(-1,1)    
    diabetes_X_test = np.empty(pred_steps)
    #incrementing index array
    
    
    for i in range(0,pred_steps):
        diabetes_X_test[i]=cols+i-1 #-1 to not coun ID col as reg counter   


    predicted_array[0] = np.around(np.concatenate((data_array[0][1:].astype(int), diabetes_X_test.astype(int))),0)
    
    
    for i in range(1, (rows-1)):
        

            
        diabetes_y = np.around(data_array[i],4)

        # Create linear regression object
        regr = linear_model.LinearRegression()

        # Train the model using the training sets
# To predict using last 3 values        
#        regr.fit(diabetes_X[-3:], diabetes_y[-3:]) 

# To predict using all last values 
        regr.fit(diabetes_X[1:], diabetes_y[1:])
        # Make predictions using the testing set
        #import pdb ;pdb.set_trace()
        diabetes_y_pred = regr.predict(diabetes_X_test.reshape(-1, 1))
        
        #switching to mnaul reg
        #diabetes_y_pred = estimate_coef(diabetes_X[-5:].flatten(), diabetes_y[-5:],(diabetes_X_test.reshape(-1, 1)))

        #Merge real values + predicted values and store to array that will be written to excel
        predicted_array[i] = np.around(np.concatenate((diabetes_y[1:], diabetes_y_pred )),4)
        
       
    #retunr [predicted_cols, apart form row 0]
    for i in range(0,pred_steps):
        predicted_step = predicted_array[1:,(cols+i-1)]
        df = pandas.read_csv(csv_file)
        header = str(df.shape[1]-1)
        df [header] = predicted_step
        df.to_csv(csv_file,index= False)
        
def estimate_coef(x, y,x_pred): 
    # number of observations/points 
    x = x.flatten()
    n = np.size(x) 
  
    # mean of x and y vector 
    m_x, m_y = np.mean(x), np.mean(y) 
  
    # calculating cross-deviation and deviation about x 
    SS_xy = np.sum(y*x) - n*m_y*m_x 
    SS_xx = np.sum(x*x) - n*m_x*m_x 
    
    # calculating regression coefficients 
    slope = SS_xy / SS_xx 
    y_intercept = m_y - slope*m_x 
    
    # Predicting further values 
    pred_y = np.empty(x_pred.shape[0])
    
    for i in range (x_pred.shape[0]):
        
        #pred_y[i] = (x_pred[i] * slope) + y_intercept

        # calc just ax instead of ax +b 
        pred_y[i] = (x_pred[i] * slope)
        
    
    return(pred_y)
