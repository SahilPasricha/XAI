import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
import sys
import os
sys.path.append(os.getcwd())
from ReadCsv.read_csv import  CsvToArray


def calc_reg(csv_file,pred_steps):
    #pred steps is number of steps to be predicted
    import pdb ;pdb.set_trace()
data_array = CsvToArray(csv_file)
    #count rows and columns
    rows = data_array.shape[0]
    cols = data_array.shape[1]


    #initialize an empty array to fill with predicted values
    predicted_array = np.empty((rows, cols))

    # Load the diabetes dataset
    #diabetes_X, diabetes_y = X[0],X[1]
    predicted_array[0] = data_array[0].astype(int)
    
    diabetes_X = data_array[0].reshape(-1,1)

    for i in range(1, (rows-1)):
        diabetes_y = np.around(data_array[i],3)
        # Use only one feature
        #diabetes_X = diabetes_X[:, np.newaxis, 2]

        # Split the data into training/testing sets
        diabetes_X_train = diabetes_X[:-pred_steps]
        diabetes_X_test = diabetes_X[-pred_steps:]

        # Split the targets into training/testing sets
        diabetes_y_train = diabetes_y[:-pred_steps]
        diabetes_y_test = diabetes_y[-pred_steps:]

        # Create linear regression object
        regr = linear_model.LinearRegression()

        # Train the model using the training sets
        regr.fit(diabetes_X_train, diabetes_y_train)

        # Make predictions using the testing set
        diabetes_y_pred = regr.predict(diabetes_X_test)

        #Merge real values + predicted values and store to array that will be written to excel
        predicted_array[i] = np.around(np.concatenate((diabetes_y_train, diabetes_y_pred )),4)

        
    import pdb;pdb.set_trace()
    print("Nothing")
