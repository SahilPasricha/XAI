import ast

from sklearn.linear_model import LinearRegression
import numpy as np
import os
import pandas as pd
# To check what all regression pt be performed and on which directory



def SimpLinearRegression(condensed_data, train_steps,predict_steps,file_path, reg_file_name):
    files = condensed_data
    trainStep = train_steps
    predStep = predict_steps
    file = condensed_data
    last_col = int(file.columns[len(file.columns)-1].replace('Epoch:',''))
    indepTrain = np.array([*(range(last_col-trainStep+1, last_col+1))])
    file_header = file.columns
    added_colList = [*range(last_col+1,last_col+1+predStep)]
    added_colList = ["Epoch:" + str(k) for k in added_colList]
    final_headers = file_header.values.tolist() + added_colList
    # DF that wil store column names and will append rows
    df_final = pd.DataFrame(columns=final_headers)



    for j in range(len(files)):
        row = file.iloc[j]
        row_val_list = row.values.tolist()
        idLessRow = row[len(file.columns)-trainStep:].to_numpy()
        model = LinearRegression()
        model.fit(indepTrain.reshape(-1, 1), idLessRow.reshape(-1, 1))
        print('coefficient of determination:', model.score(indepTrain.reshape(-1,1),idLessRow.reshape(-1,1)))
        print('intercept:', model.intercept_)
        print('slope:', model.coef_)

        indepPrd = np.asarray([*(range(last_col+1, last_col+1+predStep))])
        weightsPred = model.predict(indepPrd.reshape(-1,1))
        row_val_list.append(weightsPred[0][0])
        df_final.loc[j] = row_val_list

    #discard this file after reg is verified
    df_final.to_csv(file_path + "/" + reg_file_name)

    return (df_final,final_headers)





