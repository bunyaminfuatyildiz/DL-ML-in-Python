# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 21:30:38 2020

@author: bunya
"""

# -*- coding: utf-8 -*-
"""
Spyder Editor

Bunyamin LINEAR SVM example codes 
"""
import os
os.chdir("C:\\Users\\bunya\\python")
from egeaML import plt, train_test_split
from egeaML import functions_utils,plots,classification_plots
from sklearn.datasets.samples_generator import make_blobs
import mglearn
from sklearn.svm import SVC
import numpy as np

#understand the difference between hinge and log loss
data = np.linspace(-3,3,1000)
utils = functions_utils(data)
logistic_loss = utils.logistic_loss()
hinge_loss = utils.hinge_loss()
plots.plot_loss(data, logistic_loss, hinge_loss,
                'Logistic Loss', 'Hinge Loss','Logistic', 'Hinge'
                ,xlim=[-3,3], ylim=[-0.05, 5])

'''create a toy dataset using the scikit-learn function make_blobs: 
 that is centers= 2 will produce two different
clouds of points. We also plot the data: note that the argument s = 50 determines
the size of the balls, c = y colors the balls according to the y values'''
X, y = make_blobs(n_samples=100, centers=2,n_features=2, random_state=3, cluster_std=1.1)   
plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='winter')
plt.xlabel('Feature 0')
plt.ylabel('Feature 1')
plt.show()    

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
svc = SVC(kernel='linear')
svc.fit(X_train, y_train)
pred = svc.predict(X_test)
print(svc.score(X_test, y_test))

plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='winter')
classification_plots.plot_svc_decision_function(svc)
sv = svc.support_vectors_
sv_labels= svc.dual_coef_.ravel()>0
mglearn.discrete_scatter(sv[:,0], sv[:,1], sv_labels, s=10, markeredgewidth=1.5)
plt.xlabel('Feature 0')
plt.ylabel('Feature 1')
plt.show()

svc.support_vectors_



#GENERATE NONLINEAR DATA AND APPLY SVM WITH KERNELS.
from sklearn.datasets.samples_generator import make_circles
X, y = make_circles(100, factor=.1, noise=.1)
clf = SVC(kernel='linear').fit(X, y)
plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='winter')
classification_plots.plot_svc_decision_function(clf, plot_support=False)

clf = SVC(kernel='rbf',C=1E6)
clf.fit(X,y)
plt.scatter(X[:,0],X[:,1],c=y,s=50, cmap='winter')
classification_plots.plot_svc_decision_function(clf)
plt.scatter(clf.support_vectors_[:,0], clf.support_vectors_[:,1], s=300,lw=1, facecolors='none')
sv = clf.support_vectors_
sv_labels= clf.dual_coef_.ravel()>0
mglearn.discrete_scatter(sv[:,0], sv[:,1], sv_labels,
s=10, markeredgewidth=1.5)
plt.xlabel('Feature 0')
plt.ylabel('Feature 1')
plt.show()