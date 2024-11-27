import pandas as pd
from sklearn.model_selection import train_test_split 
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor


def selectkbest(indep_X,dep_Y,n):
    test=SelectKBest(score_func=chi2, k=n)
    fit1=test.fit(indep_X,dep_Y)
    selectK_features=fit1.transform(indep_X)
    selected_col=indep_X.columns[fit1.get_support(indices=True)]
    return selectK_features, selected_col

def split_scaler(indep_X,dep_Y):
    X_train, X_test, y_train, y_test = train_test_split(indep_X, dep_Y, test_size = 0.30, random_state = 0)
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    return X_train, X_test, y_train, y_test

def r2_prediction(regressor,X_test,y_test):
    y_pred = regressor.predict(X_test)
    r2=r2_score(y_test,y_pred)
    return r2

def Linear(X_train,y_train,X_test):
    regressor = LinearRegression()
    regressor.fit(X_train, y_train)
    r2=r2_prediction(regressor,X_test,y_test)
    return  r2

def svm_linear(X_train,y_train,X_test):
    regressor = SVR(kernel = 'linear')
    regressor.fit(X_train, y_train)
    r2=r2_prediction(regressor,X_test,y_test)
    return  r2 

def svm_NL(X_train,y_train,X_test):
    regressor = SVR(kernel = 'rbf')
    regressor.fit(X_train, y_train)
    r2=r2_prediction(regressor,X_test,y_test)
    return  r2  

def Decision(X_train,y_train,X_test):
    regressor = DecisionTreeRegressor(random_state = 0)
    regressor.fit(X_train, y_train)
    r2=r2_prediction(regressor,X_test,y_test)
    return  r2  

def random(X_train,y_train,X_test):
    regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)
    regressor.fit(X_train, y_train)
    r2=r2_prediction(regressor,X_test,y_test)
    return  r2 

def selectk_regression(acclin,accsvml,accsvmnl,accdes,accrf): 
    dataframe=pd.DataFrame(index=['ChiSquare'],columns=['Linear','SVMl','SVMnl','Decision','Random'])
    for number,idex in enumerate(dataframe.index):        
        dataframe.loc[idex,'Linear']=acclin[number]       
        dataframe.loc[idex,'SVMl']=accsvml[number]
        dataframe.loc[idex,'SVMnl']=accsvmnl[number]
        dataframe.loc[idex,'Decision']=accdes[number]
        dataframe.loc[idex,'Random']=accrf[number]
    return dataframe