import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

data = pd.read_csv('iris.csv')
data['sepal.length']= data['sepal.length'].fillna(data['sepal.length']).median()
data['sepal.width']= data['sepal.width'].fillna(data['sepal.width']).median()
X = data.drop('variety',axis =1)
y= data['variety']
from sklearn.model_selection import train_test_split 
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=2)
from sklearn.tree import DecisionTreeClassifier
dt_clf = DecisionTreeClassifier()
decision_tree_classifier=dt_clf.fit(X_train,y_train)
pickle.dump(decision_tree_classifier,open('model.pkl','wb'))