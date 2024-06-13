#import numpy as np
#import time
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score

from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
#import joblib
#from sklearn.ensemble import BaggingClassifier
import pickle

import sqlite3 
#=======================================================================
conn = sqlite3.connect('C:/Users/Atharv/Downloads/Credit_Card Adaboost Algo/Credit_Card Adaboost Algo/creditcarddb.db')


Sqlstr = "SELECT T.TCCNo,C.Id,T.TMMName,T.TMMState,T.Trans_Amt,T.TM_Status FROM Tamp_master T, card_master C WHERE T.TCCNo=C.cardno order by C.Id,T.TM_Status;"

#=======================================================================
#--------------------------------------------------------------------
df = pd.read_sql_query(Sqlstr, conn)
#print(df.head(10))


df_new=df.drop(['TCCNo'], axis=1)

df_new['TMMName'] = df['TMMName'].astype(int)
df_new['TMMState'] = df['TMMState'].astype(int)

df_new['id'] = df['id'].astype(int)
df_new['TM_Status'] = df_new['TM_Status'].astype(int)


df_new.sort_values(by=['id','TM_Status'],ascending=True,inplace=True)
#sort_by_life = gapminder.sort_values('lifeExp')
#df_new=df_new.sort_values(by=0,ascending=True,inplace=True)
#df_new.head(203)
#------------------------------------------------------------------------------

features = list(df_new[['id','TMMName','TMMState','Trans_Amt']])
target = list(df_new[['TM_Status']])


X = df_new[features] #our features that we will use to predict Y
Y = df_new[target]

#--------------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3, random_state=4)
#--------------------------------------------------------------------------------------

##########################################==================================================
####SVM Classifier#################
# def SVM_Cl():
#     C = 1.0
    
    
#     clf_SVM = svm.SVC(kernel='linear',C=C)
#     #prediction = clf_SVM.predict(X)
    
#     clf_SVM .fit(X_train, y_train.values.ravel()) 
#     prediction =clf_SVM.predict(X_test)
    
    
    
#     ## now you can save it to a file
#     with open('clf_SVM.pkl', 'wb') as f:
#         pickle.dump(clf_SVM, f)
        
# #    print("SVM Accuracy: {0:.2%}".format(accuracy_score(prediction, y_test.values.ravel())))
# #    print("Execution time: {0:.5} seconds \n".format(end-start))
    
#     A="SVM Accuracy: {0:.2%}".format(accuracy_score(prediction, y_test.values.ravel()))
#     C="SVM Model Saved as <<  clf_SVM.pkl  >>"
#     D=A+'\n'+ C
    
#     return D


def AdaBoost_Cl():
    clf_AdaBoost = AdaBoostClassifier()
    clf_AdaBoost.fit(X_train, y_train.values.ravel())
    
    prediction = clf_AdaBoost.predict(X_test)
    #scores = cross_val_score(clf_Naive, X,Y.values.ravel(), cv=5)
       
    #Predict_to_value=np.array([83.91,29,222.87])
    #Predict_to_value=Predict_to_value.reshape(1, -1)
    #Predict_get_value=clf_Naive.predict(Predict_to_value)
    #print(Predict_get_value)
           ## now you can save it to a file
    with open('clf_AdaBoost.pkl', 'wb') as f:
        pickle.dump(clf_AdaBoost, f)

    
    A="AdaBoost Accuracy: {0:.2%}".format(accuracy_score(prediction, y_test.values.ravel()))
    C="AdaBoost Model Saved as <<  clf_AdaBoost.pkl  >>"

    D = A+'\n'+ C
    return D


conn.close()

#============================================================================================