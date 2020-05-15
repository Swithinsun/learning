import numpy as py
from sklearn.cluster import KMeans

def loadData(filePath):
    fr = open(filePath,'r+')
    lines = fr.readlines()
    retData = []
    retFlowerName = []
    for line in lines:
        items = line.strip().split(",")
        print(items)
        retFlowerName.append(items[4])
        retData.append([float(items[i]) for i in range(0,len(items)-1)])
    return retData,retFlowerName

if __name__ == '__main__':
    data,FlowerName = loadData("iris.data")
    km = KMeans(n_clusters=3)
    label = km.fit_predict(data)
    result = []
    for i in range(len(label)):
        if label[i] == 0:
            result.append(1)
        elif label[i] == 1:
            result.append(2)
        else:
            result.append(3)
    print(result)
