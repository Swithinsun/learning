import pandas as pd
import numpy as np

from sklearn.metrics import classification_report

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import model_selection

def iris_type(s):
    class_label={b'Iris-setosa':0,b'Iris-versicolor':1,b'Iris-virginica':2}
    return class_label[s]

filepath = 'iris.data'
data = np.loadtxt(filepath,dtype=float,delimiter=',',converters={4:iris_type})

X,y = np.split(data,(4,),axis=1)
x_train,x_test,y_train,y_test=model_selection.train_test_split(X,y,random_state=1,test_size=0.3)

print('Start training knn')
knn = KNeighborsClassifier().fit(x_train,y_train)
print('Training Done')
answer_knn = knn.predict(x_test)
print('Prediction done')

print('Start training DT')
dt = DecisionTreeClassifier().fit(x_train, y_train)
print('Training done')
answer_dt = dt.predict(x_test)
print('Prediction done')
     
print('Start training Bayes')
gnb = GaussianNB().fit(x_train, y_train)
print('Training done')
answer_gnb = gnb.predict(x_test)
print('Prediction done')

print('\n\nThe classification report for knn:')
print(classification_report(y_test, answer_knn))
print('\n\nThe classification report for DT:')
print(classification_report(y_test, answer_dt))
print('\n\nThe classification report for Bayes:')
print(classification_report(y_test, answer_gnb))