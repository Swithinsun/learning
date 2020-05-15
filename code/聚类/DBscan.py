import numpy as np
from sklearn.cluster import DBSCAN

def load_data(data_file):
    fr = open(data_file,"r+")
    lines = fr.readlines()
    data = []
    for line in lines:
        items = line.strip().split(",")
        data.append([float(items[i]) for i in range(len(items)-1)])
    return data

if __name__ == '__main__':
    data = load_data("iris.data")
    dbs = DBSCAN(eps=0.62,min_samples=3).fit(data)
    labels = dbs.labels_
    print("Labels:")
    print(labels)