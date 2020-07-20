####################
"""
DECISION TREE (using skilearn library) QUESTION 1
SUBMITTED BY: AnkitDulani
"""
###################
 
# Importing the required packages
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.cross_validation import train_test_split
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

 
# Function importing dataset
def importdata():

    balance_data = pd.read_csv('sample.txt', sep = ',', header = None)
    # Printing the dataswet shape
    print ("Dataset Lenght: ", len(balance_data))
    print ("Dataset Shape: ", balance_data.shape)

    return balance_data
    
# Function to split the dataset
def splitdataset(balance_data):
 
    # Seperating the target variable
    X = balance_data.values[:,1:5]
    Y = balance_data.values[:,5]

    # Spliting the dataset into train and test
    X_train, X_test, y_train, y_test = train_test_split( 
    X, Y, test_size = 0.3, random_state = 1000)
     
    return X, Y, X_train, X_test, y_train, y_test
      
      
# Function to perform training with entropy.
def tarin_using_entropy(X_train, X_test, y_train):
 
    # Decision tree with entropy
    clf_entropy = tree.DecisionTreeClassifier(
            criterion = "entropy", random_state = 500,
            max_depth = 3, min_samples_leaf = 5)
 
    # Performing training
    clf_entropy.fit(X_train, y_train)
    return clf_entropy
 
# Function to make predictions
def prediction(X_test, clf_object):
 
    # Predicton on test with giniIndex
    y_pred = clf_object.predict(X_test)
    #print("Predicted values:")
    #print(y_pred)
    return y_pred
     
# Function to calculate accuracy
def cal_accuracy(y_test, y_pred):
     
    print ("Accuracy : ",
    accuracy_score(y_test,y_pred)*100)
     
# Driver code
def main():
     
    # Building Phase
    data = importdata()
    
    X, Y, X_train, X_test, y_train, y_test = splitdataset(data)
    #clf_gini = train_using_gini(X_train, X_test, y_train)
    clf_entropy = tarin_using_entropy(X_train, X_test, y_train)
     
    # Operational Phase
    print("Results Using entropy Index:")
 
    # Prediction using gini
    y_pred_gini = prediction(X_test, clf_entropy)
    cal_accuracy(y_test, y_pred_gini)
    
# Calling main function
if __name__=="__main__":
    main()