# -*- coding: utf-8 -*-
from __future__ import print_function

"""
Created on Sun Oct  4 15:20:58 2020

@author: bunyaminfuatyildiz
"""
''' Lets  apply  knn'''

#det wd
import os
os.chdir('c:\\Users\\bunya\\python')     



#import libraries we need

from egeaML import *
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import scale
import pickle

#Use tensorflow backend
import warnings
warnings.filterwarnings('ignore')
# We import and read the data,
reader = DataIngestion(df='data_intro.csv', col_target='male')
'''
  This class is used to ingest data into the system before preprocessing.
  It reads the data from a .csv ﬁle; It split the data into features and target, denoted respectively by X and y.
    '''
data = reader.load_data()
X = reader.features()
y = reader.target()

''' test and training must be kept independent from each other.
To do this, we use the scikit-learn method train_test_split from the model_selection
module, which requires the user to specify the percentage of the available data to be
used for the test set.'''
#following snippet produce a 2-dimensional plot showing the relationship between height and weight, marked by their corresponding label
classification_plots.training_class(X,y,test_size=0.3)
'''This plot shows the relationship between the two-dimensional, real-valued training
dataset, and that there are two classes by which it is possible to split the data. '''
''' Let’s see how Nearest Neighbors
works in practice, using the standard scikit pipeline.'''
X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.3, random_state=42)
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train,y_train)
y_pred = knn.predict(X_test)
score = knn.score(X_test,y_test)
print("accuracy: {:.4f}".format(score))
'''We now plot the predicted labels, using the egeaML library, highlight-
ing the ones who were uncorrectly classiﬁed by our model:'''
classification_plots.plotting_prediction(X_train,X_test,y_train,y_test,nn=1)
''' Remark:In many situation, we train a model with tons of examples. it is good practice to store the ﬁtted model into a pickle ﬁle
A possible use of a pickle is to keep track of the ﬁtted model as soon as a new retraining happens.'''
pkl_filename = "my_first_ML_model.pkl"
with open(pkl_filename, 'wb') as file:
    pickle.dump(knn, file)

'''confusion
matrix, which diagonal elements represent the true negative (TN) - that is examples
that have been predicted as female and are indeed female - and true positive (TP) -
that is examples who are men and the model predicted them as men -, respectively'''
classification_plots.confusion_matrix(y_test,y_pred)

'''well done for now but what happens if we increase the number of
neighbors? The next chunk produces a plot that shows the accuracy of the model
for different values of the hyperparameter n_neighbors:'''
n_neigh = list(range(1,50))
train_scores = []
test_scores = []
for i in n_neigh:
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,y_train)
    train_score = knn.score(X_train,y_train)
    train_scores.append(train_score)
    test_score = knn.score(X_test,y_test)
    test_scores.append(test_score)
    
df = pd.DataFrame()
df['n_neigh']= n_neigh
df['Training Score']=train_scores
df['Test Score']=test_scores
plt.figure(figsize=(5,5))
plt.plot(df.iloc[:,0], df.iloc[:,1], label ='Train Performance')
plt.plot(df.iloc[:,0], df.iloc[:,2], label ='Test Performance')
plt.xlabel('Number of Neighbors', fontsize=16)
plt.ylabel('Accuracy', fontsize=16)
plt.legend()
plt.show()    

#Tuning Hyperparameters with Cross-Validation
''' you will use validation sets to tune hyper parameters. lets use better dataset (brest cancer wisconsin)'''

data_ = DataIngestion(df='breast_cancer_data.csv',col_to_drop=None,col_target='diagnosis')
X = data_.features()
y = data_.target().apply(lambda x: 1 if x=='M' else 0)
X=scale(X)                    
'''We now split the data into training, validation and test set'''
X_train, X_test, y_train, y_test= train_test_split(X,y,test_size=0.3,random_state=42)
X_train_,X_val,y_train_,y_val = train_test_split(X_train, y_train,test_size=0.3, random_state=42)
knn = KNeighborsClassifier(n_neighbors=5).fit(X_train_,y_train_)
print("Validation Score: {:.4f}".format(knn.score(X_val,y_val)))
print("Test Score: {:.4f}".format(knn.score(X_test,y_test)))
X_train,X_test, y_train, y_test = train_test_split(X,y,test_size=0.3, random_state=42)
cross_val_scores = []
neighbors = np.arange(1,15,2)
for i in neighbors:
    knn = KNeighborsClassifier(n_neighbors=i)
    scores = cross_val_score(knn,X_train,y_train,cv=5)
    cross_val_scores.append(np.mean(scores))
print("Best CV Score: {:.4f}".format(np.max(cross_val_scores)))
best_nn = neighbors[np.argmax(cross_val_scores)]
print("Best n_neighbors: {}".format(best_nn))

'''IMPLEMENTING GRID SEARCH FOR CV '''
from sklearn.model_selection import GridSearchCV
X_train, X_test, y_train, y_test = train_test_split(X,y,stratify=y,test_size=0.3,random_state=42)
param_grid = {'n_neighbors':np.arange(1,15,2)}
clf = KNeighborsClassifier()
grid = GridSearchCV(clf, param_grid= param_grid, cv=10)
grid.fit(X_train,y_train)
print("Best Mean CV Score: {:.4f}".format(grid.best_score_))
print("Best Params: {}".format(grid.best_params_))
print("Test-set Score: {:.4f}".format(grid.score(X_test,y_test)))

