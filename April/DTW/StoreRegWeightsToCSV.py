'''
Input -> original weights file , an df[regression columns, clustered ids]
output --> Excel with weights + regresion weights
Logic- The weights were clustered before REG,
now we need to feed groups with regression weigts
How to know which weights to which id --> Clustered id contains aggregate of ids to whom these weights belong
'''

def StoreWeights():
