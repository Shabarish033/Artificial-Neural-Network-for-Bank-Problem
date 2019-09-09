# -*- coding: utf-8 -*-
#%% Importing Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, OneHotEncoder #Used in data preprocessing for encoding input data
from sklearn.model_selection import train_test_split #Data Preprocessing - Splitting data set
from sklearn.preprocessing import StandardScaler #Standardizing data
import keras
from keras.models import Sequential #ANN Builders
from keras.layers import Dense #Defining Layers within the ANN
from sklearn.metrics import confusion_matrix #To compute the efficiency of predictions
from keras.wrappers.scikit_learn import KerasClassifier #Links Sklearn and keras
from sklearn.model_selection import cross_val_score #Measures the accuracies
from sklearn.model_selection import GridSearchCV

# Data Preprocessing
#Reading Inputs
data = pd.read_csv('Churn_Modelling.csv')
X = data.iloc[:,3:13]
Y = data.iloc[:,-1]

#Encoding Columns of data
LabelEncoder_X1 = LabelEncoder()
X.iloc[:,1] = LabelEncoder_X1.fit_transform(X.iloc[:,1])
LabelEncoder_X2 = LabelEncoder()
X.iloc[:,2] = LabelEncoder_X2.fit_transform(X.iloc[:,2])
onehotencoder = OneHotEncoder(categorical_features=[1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:] #This is done to avoid dummy variable trap

#Test and Training datasets
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X,Y, test_size=1/5, random_state=0)

# Scaling data
ss = StandardScaler()
Xtrain = ss.fit_transform(Xtrain)
Xtest = ss.fit_transform(Xtest)

# Creating the ANN
classifier = Sequential()
#1st Input Layer
classifier.add(Dense(units=6, kernel_initializer='glorot_uniform', input_dim=11))
#2nd Input Layer
classifier.add(Dense(units=6, activation='relu', kernel_initializer='glorot_uniform'))
#Output Layer
classifier.add(Dense(units=1, activation='sigmoid', kernel_initializer='glorot_uniform'))
#Compiling
classifier.compile(optimizer='adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
#Fitting the Neural Network
classifier.fit(Xtrain, Ytrain, batch_size=10,epochs=100)
# Part 3 - Making predictions and evaluating the model

# Predicting the Test set results
Ypred = classifier.predict(Xtest)
Ypred = (Ypred > 0.5)

# Improving of ANN - KFold Cross validation
# Tuning Parameters
def Classifier_build():
    classifier = Sequential()
    #1st Input Layer
    classifier.add(Dense(units=6, kernel_initializer='glorot_uniform', activation='relu', input_dim=11))
    #2nd Input Layer
    classifier.add(Dense(units=6, kernel_initializer='glorot_uniform',activation='relu'))
    #Output Layer
    classifier.add(Dense(units=1, kernel_initializer='glorot_uniform', activation='sigmoid'))
    #Compiling
    classifier.compile(optimizer='adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier

classifier = KerasClassifier(build_fn = Classifier_build, batch_size=10, epochs=100)
accuracies = cross_val_score(estimator = classifier, X = Xtrain, y = Ytrain, cv = 10, n_jobs = -1) #
mean = accuracies.mean() #Find the mean value of accuracies
variance = accuracies.std() #Find the variance of the accuracies
    
# Parameter Tuning using GridSearch
def Classifier_build(optimizer):
    classifier = Sequential()
    #1st Input Layer
    classifier.add(Dense(units=6, kernel_initializer='glorot_uniform', activation='relu', input_dim=11))
    #2nd Input Layer
    classifier.add(Dense(units=6, kernel_initializer='glorot_uniform',activation='relu'))
    #Output Layer
    classifier.add(Dense(units=1, kernel_initializer='glorot_uniform', activation='sigmoid'))
    #Compiling
    classifier.compile(optimizer=optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier

classifier = KerasClassifier(build_fn = Classifier_build, batch_size=10, epochs=100)

#Create a dictionary for the parameters
parameters = {'batch_size': [15, 25, 32], 'epochs':[100, 500], 'optimizer':['adam', 'rmsprop']}
gridSearch = GridSearchCV(estimator=classifier, param_grid = parameters, scoring='accuracy', cv=10)
gridSearch = gridSearch.fit(Xtrain, Ytrain)
bestparameters = gridSearch.best_params_
bestaccuracy = gridSearch.best_score_

    
    