import pandas as pd
from sklearn.model_selection import train_test_split
import time
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


def selectkbest(indep_X,dep_Y,n):
    test=SelectKBest(score_func=chi2, k=n)
    fit1=test.fit(indep_X,dep_Y)
    selectK_features=fit1.transform(indep_X)
    selected_col=indep_X.columns[fit1.get_support(indices=True)]
    return selectK_features,selected_col

def split_scaler(indep_X,dep_Y):
    X_train, X_test, y_train, y_test = train_test_split(indep_X, dep_Y, test_size = 0.30, random_state = 0)
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    return X_train, X_test, y_train, y_test

def cm_prediction(classifier,X_test):
    y_pred = classifier.predict(X_test)

    # Making the Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    
    Accuracy=accuracy_score(y_test, y_pred )

    report=classification_report(y_test, y_pred)
    return  classifier,Accuracy,report,X_test,y_test,cm

def logistic(X_train,y_train,X_test):
    classifier = LogisticRegression(random_state = 0)
    classifier.fit(X_train, y_train)
    classifier,Accuracy,report,X_test,y_test,cm=cm_prediction(classifier,X_test)
    return  classifier,Accuracy,report,X_test,y_test,cm

def svm_linear(X_train,y_train,X_test):
    classifier = SVC(kernel = 'linear', random_state = 0)
    classifier.fit(X_train, y_train)
    classifier,Accuracy,report,X_test,y_test,cm=cm_prediction(classifier,X_test)
    return  classifier,Accuracy,report,X_test,y_test,cm

def svm_NL(X_train,y_train,X_test):
    classifier = SVC(kernel = 'rbf', random_state = 0)
    classifier.fit(X_train, y_train)
    classifier,Accuracy,report,X_test,y_test,cm=cm_prediction(classifier,X_test)
    return  classifier,Accuracy,report,X_test,y_test,cm

def Navie(X_train,y_train,X_test):     
    classifier = GaussianNB()
    classifier.fit(X_train, y_train)
    classifier,Accuracy,report,X_test,y_test,cm=cm_prediction(classifier,X_test)
    return  classifier,Accuracy,report,X_test,y_test,cm 

def knn(X_train,y_train,X_test):
    classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
    classifier.fit(X_train, y_train)
    classifier,Accuracy,report,X_test,y_test,cm=cm_prediction(classifier,X_test)
    return  classifier,Accuracy,report,X_test,y_test,cm

def Decision(X_train,y_train,X_test):
    classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
    classifier.fit(X_train, y_train)
    classifier,Accuracy,report,X_test,y_test,cm=cm_prediction(classifier,X_test)
    return  classifier,Accuracy,report,X_test,y_test,cm    

def random(X_train,y_train,X_test):
    classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
    classifier.fit(X_train, y_train)
    classifier,Accuracy,report,X_test,y_test,cm=cm_prediction(classifier,X_test)
    return  classifier,Accuracy,report,X_test,y_test,cm

def selectk_Classification(acclog,accsvml,accsvmnl,accknn,accnav,accdes,accrf):
    dataframe=pd.DataFrame(index=['ChiSquare'],columns=['Logistic','SVMl','SVMnl','KNN','Navie','Decision','Random'])
    for number,idex in enumerate(dataframe.index):      
        dataframe.loc[idex,'Logistic']=acclog[number]       
        dataframe.loc[idex,'SVMl']=accsvml[number]
        dataframe.loc[idex,'SVMnl']=accsvmnl[number]
        dataframe.loc[idex,'KNN']=accknn[number]
        dataframe.loc[idex,'Navie']=accnav[number]
        dataframe.loc[idex,'Decision']=accdes[number]
        dataframe.loc[idex,'Random']=accrf[number]
    return dataframe