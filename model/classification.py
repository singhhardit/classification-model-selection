# coding: utf-8
# NOTE THIS SCRIPT IS WRITTEN FOR python2.7

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
import random

# Assumes inputs are pandas data frames
# Assumes the last column of data is the output dimension

# Default Model
def get_pred_default(train,test):
    n_train,m_train = train.shape
    n_test,m_test=test.shape
    def_pred = [train.iloc[:,-1].value_counts().idxmax()]*n_test
    true = test.iloc[:,-1]
    pred=pd.DataFrame({'pred':def_pred,
                       'true':true })
    return pred

# Logistic Regression
# Assumes the last column of data is the output dimension
def get_pred_logreg(train,test):
    n,m = train.shape # number of rows and columns
    x_train=train.iloc[:,:m-1] #assumes last columns as output variable
    y_train=train.iloc[:,-1] #predictor variable of training set
    x_test=test.iloc[:,:m-1] #testing set
    y_test=test.iloc[:,-1]
    regressor = LogisticRegression()  
    regressor.fit(x_train, y_train)  
    y_pred = regressor.predict_proba(x_test)[:,1]
    pred=pd.DataFrame({'pred':y_pred,
                       'true':y_test })
    return pred

# Support Vector Machine
# Assumes the last column of data is the output dimension
def get_pred_svm(train,test):
    n,m = train.shape # number of rows and columns
    x_train=train.iloc[:,:m-1] #assumes last columns as output variable
    y_train=train.iloc[:,-1] #predictor variable of training set
    x_test=test.iloc[:,:m-1] #testing set
    y_test=test.iloc[:,-1]
    model = svm.SVC(kernel='rbf', probability=True) # Linear Kernel
    model.fit(x_train, y_train)
    
    #Predict the response for test dataset
    y_pred = model.predict_proba(x_test)[:,1]
    pred=pd.DataFrame({'pred':y_pred,
                       'true':y_test })
    return pred

# Naive Bayes
# Assumes the last column of data is the output dimension
def get_pred_nb(train,test):
    n,m = train.shape # number of rows and columns
    x_train=train.iloc[:,:m-1] #assumes last columns as output variable
    y_train=train.iloc[:,-1] #predictor variable of training set
    x_test=test.iloc[:,:m-1] #testing set
    y_test=test.iloc[:,-1]
    model = GaussianNB()
    model.fit(x_train, y_train)
    y_pred=model.predict_proba(x_test)[:,1]
    pred=pd.DataFrame({'pred':y_pred,
                       'true':y_test })
    return pred

# k-Nearest Neighbor
# Assumes the last column of data is the output dimension
def get_pred_knn(train,test,k):
    
    n,m = train.shape # number of rows and columns
    x_train=train.iloc[:,:m-1] #assumes last columns as output variable
    y_train=train.iloc[:,-1] #predictor variable of training set
    x_test=test.iloc[:,:m-1] #testing set
    y_test=test.iloc[:,-1]
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(x_train, y_train)
    y_pred=model.predict_proba(x_test)[:,1]
    pred=pd.DataFrame({'pred':y_pred,
                       'true':y_test })
    return pred

#k fold cross validation implementation
def do_cv_class(df, num_folds, model_name):
    n,m=df.shape
    rand_index=random.sample(range(n), n) #taking random sample indexes
    chunks=[rand_index[i::num_folds] for i in xrange(num_folds)] #dividing random indexes into k chunks
    score=['NaN']*len(chunks) #initialising score to be returned
    for i in range(0,len(chunks)):
        test_index=chunks[i]    #one chunk becomes test
        train_index=list(set(rand_index) - set(test_index)) #remaining chunks become train
        test=df.iloc[test_index,:]
        train=df.iloc[train_index,:]
        if(model_name is 'logreg'):
            prediction=get_pred_logreg(train.select_dtypes(include=[np.number]), #excluding symbolic attributes
                         test.select_dtypes(include=[np.number]))
            prediction['fold']=i+1
        elif(model_name is 'svm'):
            prediction=get_pred_svm(train.select_dtypes(include=[np.number]), #excluding symbolic attributes
             test.select_dtypes(include=[np.number]))
            prediction['fold']=i+1
        elif(model_name is'nb'):
            prediction=get_pred_nb(train.select_dtypes(include=[np.number]), #excluding symbolic attributes
             test.select_dtypes(include=[np.number]))
            prediction['fold']=i+1
        elif('nn' in model_name):
            prediction=get_pred_knn(train.select_dtypes(include=[np.number]), #excluding symbolic attributes
             test.select_dtypes(include=[np.number]),
             int(filter(str.isdigit, model_name)))
            prediction['fold']=i+1
        else :
            prediction=get_pred_default(train.select_dtypes(include=[np.number]), #excluding symbolic attributes
             test.select_dtypes(include=[np.number]))
            prediction['fold']=i+1
        score[i]=prediction #initialising score to be returned
    result=pd.concat(score)    
    return result

#input prediction file the first column of which is prediction value
#the 2nd column is true label (0/1)
#cutoff is a numeric value, default is 0.5
def get_metrics(pred, cutoff=0.5):
    pred.iloc[:,0] = np.where(pred.iloc[:,0]> cutoff, 1, 0)
    df_confusion = pd.crosstab(pred.iloc[:,1], pred.iloc[:,0], rownames=['Actual'], colnames=['Predicted'], margins=True)
    fp = df_confusion.iloc[1,0]    
    fn = df_confusion.iloc[0,1]    
    tp = df_confusion.iloc[0,0]    
    tn = df_confusion.iloc[1,1]
    # Sensitivity, hit rate, recall, or true positive rate
    metrics = pd.DataFrame({
            'TPR': tp/float(tp+fn),
    # Fall out or false positive rate
            'FPR' : fp/float(fp+tn),
    # Overall accuracy
            'ACC' : (tp+tn)/float(tp+fp+fn+tn),
    #Precision 
            'PRE' : (tp)/float(tp+fp),            
    #Recall
            'REC' : tp/float(tp+fn)}, index=[0])
    return metrics

####import data#####

my_data = pd.read_csv('wine.csv')
#encode class into 0/1 for easier handling by classification algorithm
my_data['type'] = np.where(my_data['type'] == 'high', 1, 0)
#print get_metrics(result)

def print_cont_table(df):
    df['pred'] = np.where(df['pred'] > 0.5, 1, 0)
    df_confusion = pd.crosstab(df.iloc[:,1], df.iloc[:,0], rownames=['Actual'], colnames=['Predicted'], margins=True)
    print df_confusion

#(a) finding the parameter k in the kNN model that gives the best generalization
error=[]
kval=[]
acc=[]
#considering k between 1 - 25
for k in range(1,25):
    tmp = do_cv_class(my_data,10,str(k)+'nn')
    metrics=get_metrics(tmp.iloc[:, 0:2])
    error.append(1-metrics.iloc[0][0])
    acc.append(metrics.iloc[0][0])
    kval.append(k)
calc_error=pd.DataFrame({'k': kval,'err': error})

print 'The value of k that gives best generalization and its accuracy:'
print [acc.index(max(acc))+1, round(max(acc),4)]

#plotting error on test set 
error_plot = calc_error.plot.line(x='k', y='err', style='.-')

print 'according to the learning curve, K=6 gives the best fit'
print 'K < 6 leads to overfitting'
print 'K > 6 leads to underfitting'

# (b)

print '--------------------'
print 'default'
print '--------------------'
tmp = do_cv_class(my_data,10,'default') # returns pandas dataframe
print get_metrics(tmp.iloc[:, 0:2])


print '-------------------'
print 'logistic regression'
print '-------------------'
tmp = do_cv_class(my_data,10,'logreg') # returns pandas dataframe
print get_metrics(tmp.iloc[:, 0:2])


print '--------------------'
print 'naieve Bayes'
print '--------------------'
tmp = do_cv_class(my_data,10,'nb') # returns pandas dataframe
print get_metrics(tmp.iloc[:, 0:2])

print '--------------------'
print 'svm'
print '--------------------'
tmp = do_cv_class(my_data,10,'svm') # returns pandas dataframe
print get_metrics(tmp.iloc[:, 0:2])


print '--------------------'
print 'KNN'
print '--------------------'
tmp = do_cv_class(my_data,10,'6nn') # returns pandas dataframe
print get_metrics(tmp.iloc[:, 0:2])
