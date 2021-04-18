import ast
from sklearn.linear_model import LinearRegression
import numpy as np
import os
import pandas as pd

def storeWeights(dfReg):
    Origfile = "/home/sap/IdeaProjects/XAI/April/weights/Rweights00.csv"
    origWeights = pd.read_csv(Origfile)
    lastCol = int(origWeights.columns[-1])
    for i in range(len(dfReg.columns[1:])):
        origWeights[str(lastCol+1+i)] = ''
    for i in range(len(dfReg)):
        #temp
        #import pdb;pdb.set_trace()
        # counter of number of rows in condensed weights
        curIdList = ast.literal_eval(dfReg.iloc[i]['GroupedIds'])
        clusterRegWeights = dfReg.iloc[i]
        for j in curIdList:
            #import pdb;pdb.set_trace()
            idx = (origWeights[origWeights['Id'] == j].index).to_numpy()[0]
            for k in (dfReg.columns[1:]):
                origWeights.at[idx, str(k)] = clusterRegWeights[k]

    import pdb;pdb.set_trace()
    print("abc")



#def RegressionMain(file_path, trainStep, predStep, linear = False, self=None):
def RegressionMain(file_path= "/home/sap/IdeaProjects/XAI/April/weights", trainStep = 3, predStep = 3,self=None):
    self.file_path = file_path
    #self.linear = linear
    allFiles = os.list(file_path)
    filteredFiles = [i for i in allFiles if i.startswith('clustered')]




#To run from console, remove below line after it

#def LinearRegression(file_path1 ,files , trainStep,predStep):
file_path1 = "/home/sap/IdeaProjects/XAI/April/weights/"
files = ['test2.csv']
trainStep = 3
predStep = 3


print("Inside")
for i in range(len(files)):
    import pdb;pdb.set_trace()
    file = pd.read_csv(file_path1 + files[i])
    last_col = int(file.columns[len(file.columns)-1])
    indepTrain = (file.columns[len(file.columns)-trainStep:]).to_numpy()
    depPred = []
    #working to get last col and use it to start point of predict
    #dataRaw = pd.read_csv(file)
    for j in range(len(file)):
        row = file.iloc[j]
        #using last trainStep values for train
        idLessRow = row[len(file.columns)-trainStep:].to_numpy()

        model = LinearRegression()
        model.fit(indepTrain.reshape(-1, 1), idLessRow.reshape(-1, 1))
        print('coefficient of determination:', model.score(indepTrain.reshape(-1,1),idLessRow.reshape(-1,1)))
        print('intercept:', model.intercept_)
        print('slope:', model.coef_)

        indepPrd = np.asarray([*(range(last_col+1, last_col+1+predStep))])
        weightsPred = model.predict(indepPrd.reshape(-1,1))

        colList = [*range(last_col+1,last_col+1+predStep)]
        colList.insert(0,'GroupedIds')
        clusteredIds = file.iloc[j]['GroupedIds']
        weightPrdList = weightsPred.tolist()
        weightPrdList.insert(0,clusteredIds)


        if j == 0:
            dfReg = pd.DataFrame([weightPrdList], columns=colList)

        else:
            dfReg = dfReg.append(pd.DataFrame([weightPrdList], columns=colList))
    clusteregReg = (pd.merge(file, dfReg, on='GroupedIds'))
    clusteregReg.to_csv(file_path1 + "DoneTest212.csv")
    # For now, save this file with new name and compare to original file ,
    # later overrise the same file
    import pdb;pdb.set_trace()
    print("12")
    storeWeights(dfReg)