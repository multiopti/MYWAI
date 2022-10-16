# -*- coding: utf-8 -*-
"""
Created on Sun Oct  16 

@author: GustavoSanchez
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.datasets import make_classification
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.preprocessing import StandardScaler
import csv

np.random.seed(42)

def make_moons_ndim(
    ndim, 
    radius, 
    number_of_datapoints,
):
    
    X = np.zeros([number_of_datapoints,ndim])
    S = np.zeros(number_of_datapoints*ndim)
    Y = np.zeros(number_of_datapoints)
    for i in range(number_of_datapoints):
        q = np.random.uniform(-1,1,size=ndim)
        q = radius*np.sqrt(ndim)*(1/np.linalg.norm(q))*q
        S[i*ndim:(i+1)*ndim] = q
        if q[0]*q[-1] >= 0:
           y = 0
        else:
           y = 1

        X[i,:] = q
        Y[i] = y

    return X, Y, S


X, y, S = make_moons_ndim(
    ndim=16,
    radius=1, 
    number_of_datapoints=1000)

two_moons = (X, y)

X, y = make_classification(
    n_features=16, n_redundant=14, n_informative=2, random_state=1, n_clusters_per_class=1
)
rng = np.random.RandomState(42)
X += 2 * rng.uniform(size=X.shape)
sklearn_bench = (X, y)

names = [
    "Nearest Neighbors",
    "RBF SVM",
    "Gaussian Process",
    "Decision Tree",
    "Random Forest",
    "Neural Net",
    "AdaBoost",
    "Naive Bayes",
    "QDA",
]

classifiers = [
    KNeighborsClassifier(3),
    SVC(gamma=1, C=1),
    GaussianProcessClassifier(1.0 * RBF(1.0)),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    MLPClassifier(alpha=1, max_iter=1000),
    AdaBoostClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis(),
]

datasets = [
    two_moons,
    sklearn_bench,
]


# iterate over dataset
score_table = []
for ds_cnt, ds in enumerate(datasets):
    # preprocess dataset, split into training and test part
    X, y = ds
    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

        # iterate over classifiers

    score  = 0
    for name, clf in zip(names, classifiers):
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)
        # print('dataset = ', ds_cnt)
        # print('classifier = ', clf)
        # print('score = ', score)
        score_table.append([ds_cnt,name,score])
        score  = 0
print(score_table)



with open('C:\\Users\\GustavoSanchez\\score_table', 'w') as f:
    write = csv.writer(f)    
    write.writerows(score_table)