####################
"""
DECISION TREE (using Binary tree as data structure) QUESTION 2
SUBMITTED BY: AnkitDulani
"""
#####################
import numpy as np
import pandas as pd
import csv
from sklearn.metrics import accuracy_score
import math
import random

#   DATASTRUCTURE TO STORE DATA
class BinaryTree():

    def __init__(self,data=None):
      self.left = None
      self.right = None
      self.attr= data
      self.leaf=True
      self.value=0
      self.index=-1

    def getLeftChild(self):
        return self.left
    def getRightChild(self):
        return self.right
    def setNodeAttr(self,value):
        self.attr = value
    def getNodeAttr(self):
        return self.attr
    def appendR(self,value):
        self.leaf=False
        self.right=value
    def appendL(self,value):
        self.leaf=False
        self.left=value
    def isLeaf(self):
        return self.leaf
    def setValue(self,value):
        self.value=value
    def getValue(self):
        return self.value
    def setIndex(self,value):
        self.index= value
    def getIndex(self):
        return self.index
    def setAsLeaf(self):
        self.leaf=True
    def resetAsNode(self):
        self.leaf=False
#   FUNCTION TO PRINT BINARY TREE
def printT(root):

    if root.getLeftChild() != None:
        printT(root.getLeftChild())
    print(root.getIndex())
    if root.getRightChild() != None:
        printT(root.getRightChild())
#   CALCULATE LOG WITH BASE 2
def log2(value):
    if value is 0:
        return 0
    else:
        return math.log(value)/math.log(2)

#   TO CALCULATE ENTROPY
def calcEntropy(Y):
 
    pos=0
    neg=0
    for i in range(len(Y)):
        if Y[i] == 1:
            pos+=1
        else:
            neg+=1

    tot=pos+neg
    if neg==0 or pos==0:
        return 0
    else:
        return -1*((pos/tot)*log2(pos/tot)+(neg/tot)*log2(neg/tot))

# Fn TO CALCULATE INFORMATION GAIN
def informationGain(X,Y,entropy):
    a=[]
    b=[]
    count=0
    for i in range(len(X)):
        if X[i]==1:
            a.append(Y[i])
            count+=1
        else:
            b.append(Y[i])
    total=len(X)
    return entropy-(count*calcEntropy(a)+(total-count)*calcEntropy(b))/total

# Fn TO SELCT ATTRIBUTE
def selectattribute(dataset,Y,xlist):
    entropy=calcEntropy(Y)
    if entropy is 0:
        return -1

    minGain=0
    attr=-1

    for i in range(len(dataset[0])):
        if i in xlist:
            continue
        a=informationGain(dataset[:,i],Y,entropy)
        if minGain <= a:
            minGain=a
            attr=i

    return attr

# Fn TO BUILDTREE
def buildTree(dataset,xlist):

    #xlist store the attribute which are ancestor to the particular node
    if len(xlist)== 20:
        return BinaryTree()

    attr=selectattribute(dataset.values[:,0:-1],dataset.values[:,-1],xlist)
   
    # intialisation of binary tree
    root=BinaryTree()
    
    # if IG=1 thereby we conclude it as leaf
    if attr is -1:
        if dataset.values[0][-1] == 1:
            root.setValue(1)
        else:
            root.setValue(0)
        return root

    #adding attr to xlist
    xlist.append(attr)
    root.setNodeAttr(attr)

    # intialisation of new dataset based upon classification by an attribute
    leftlist=dataset.head(0)
    rightlist=dataset.head(0)

    # creating a list of classified dataset accor. to attribute
    for i in range(len(dataset)):
        if dataset.values[i][attr] == 1:
            leftlist=leftlist.append(dataset[i:i+1],ignore_index=True)
        else:
            rightlist=rightlist.append(dataset[i:i+1],ignore_index=True)

    # callingg build function for it child tree
    root.appendL(buildTree(leftlist,xlist))
    root.appendR(buildTree(rightlist,xlist))


    for i in range(len(xlist)):
        if xlist[i]==attr:
            del xlist[i]
            break

    #"""
    del leftlist
    del rightlist
    #"""
    return root

 ############ Setting node as a leaf

# assigning Index to the node of Tree
def assignIndex(root,nodeIndex):

    if root==None:
        return 
    if root.isLeaf() ==True:
        return 
    assignIndex(root.getLeftChild(),nodeIndex)
    root.setIndex(nodeIndex[0])
    nodeIndex[0]+=1
    assignIndex(root.getRightChild(),nodeIndex)

# function to find a particulare node in Tree
def findNode(root,n):

    if root.getIndex() == n:
        return root
    if root.getIndex()>=n:
        return findNode(root.getLeftChild(),n) 
    else:
        return findNode(root.getRightChild(),n)

# function to count the classfied data
def countLeaf(node):
    a=0
    if node.isLeaf() ==True:
        a= node.getValue()
        if a == True:
            return 1
        else:
            return -1

    a+=countLeaf(node.getRightChild())
    a+=countLeaf(node.getRightChild())
    return a

#function used in pruning
def setNodeAsLeaf(root,n):
    node=findNode(root,n)
    pos=countLeaf(node)
    node.setAsLeaf()
    if pos > 0:
        node.setValue(1)
    else:
        node.setValue(0)
    return node

# importing data to the function
def importdata():
    
    test_data = pd.read_csv('test_set.csv', sep = ',')
    validation_data = pd.read_csv('validation_set.csv', sep = ',')
    train_data = pd.read_csv('training_set.csv', sep = ',')
    """
    with open('training_set.csv') as csvfile:
        rows = csv.reader(csvfile)
        train_data = list(zip(*rows))
    with open('test_set.csv') as csvfile:
        ro = csv.reader(csvfile)
        test_data= list(zip(*ro))
    with open('validation_set.csv') as csvfile:
        roq = csv.reader(csvfile)
        validation_data= list(zip(*roq))
    """
    return test_data,train_data,validation_data

def splitdataset(balance_data):
    # Seperating the target variable
    X = balance_data.values[0:,0:-1]
    Y = balance_data.values[:,-1]
    return X, Y

def predict(X,root):
    if root.isLeaf() == True:
        return root.getValue()
    if X[root.getNodeAttr()] == 1:
        return predict(X,root.getLeftChild())
    else:
        return predict(X,root.getRightChild())

def prediction(dataset,root):
    X,Y=splitdataset(dataset)
    Y_pred=[]
    for i in range(len(Y)):
        Y_pred.append(predict(X[i],root))

    return Y_pred

def main():
     
    # Building Phase
    test_data,train_data,validation_data = importdata()

    #Splitting the dataSet
    
    test_data_X,test_data_Y=splitdataset(test_data)
    train_data_X,train_data_Y=splitdataset(train_data)
    validation_data_X,validation_data_Y=splitdataset(validation_data)

   
    excludeList=[]
    root=buildTree(train_data,excludeList)

    nodeIndex=[0]
    assignIndex(root,nodeIndex)
 
    Y_pred=prediction(validation_data,root)
    Y_test=prediction(test_data,root)
    ValidationAccurcayBeforePruning=accuracy_score(Y_pred,validation_data_Y)*100
    TestAccurcayBeforePruning=accuracy_score(Y_test,test_data_Y)*100
    valAcc=ValidationAccurcayBeforePruning
    print("Validation Accuracy Before Pruning : ",ValidationAccurcayBeforePruning)
    print("Accuracy for Before Prunning: ",TestAccurcayBeforePruning)

    #printT(root)
    print("Enter value of l and K above 25 to get best Decision Tree")
    L=int (input("enter value for L: "))
    K=int (input("enter value for K: "))

    ####### Pruning of the decision Tree
    n=int(nodeIndex[0])
    for i in range(L):
        m=random.randint(1,K)
        for j in range(m):
            p= random.randint(0,n-1)
            Troot=setNodeAsLeaf(root,p)
            Ypric=prediction(validation_data,root)
            acc=accuracy_score(Ypric,validation_data_Y)*100
            if acc >= valAcc:
                valAcc=acc
            else:
                Troot.resetAsNode()



    print("Validation Accuracy After Pruning : ",valAcc)
    testacc=accuracy_score(test_data_Y,prediction(test_data,root))*100
    print("Accuracy for After Prunning: ",testacc)

# Calling main function
if __name__=="__main__":
    main()