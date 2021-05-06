'''
To uncluster weights that ere clustered for regression
Output - same as weights before cluster with regression values added in last
'''

import pandas as pd

def uncluster_reg_weights(reg_weights,headers,file_path,file_name):
    pd.options.mode.chained_assignment = None
    df_unclustered = pd.DataFrame(columns=headers)
    counter = 0
    for i in range(len(reg_weights)):
        clustered_id = reg_weights.iloc[i]['Id']
        for j in clustered_id:
            idless_weights = (reg_weights.loc[i][1:]).values.tolist()
            idless_weights.insert(0, j)
            df_unclustered.loc[counter] = idless_weights
            counter += 1
    df_sorted = df_unclustered.sort_values(by=['Id'])
    df_sorted.to_csv(file_path + "/" + file_name)
    return df_sorted






