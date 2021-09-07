#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from IPython.display import Image


# In[3]:


import warnings
warnings.filterwarnings("ignore") # 把warning關掉


# In[4]:


import sys
print (sys.version) 
print (sys.version_info)


# In[5]:


from sklearn import datasets
from sklearn.preprocessing import StandardScaler 
from sklearn.datasets import load_breast_cancer
from sklearn.pipeline import make_pipeline,Pipeline

from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold, train_test_split
from sklearn.model_selection import learning_curve, validation_curve, cross_val_score

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.svm import SVC

from sklearn.ensemble import VotingClassifier

from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score

import sklearn.externals
import joblib


# In[6]:


#讀取
data = pd.read_csv('0717MLdatacsv.csv')
data.head()


# In[7]:


# 編碼
classnumlize = {'Source.Name':{'B': 0, 'R': 1, 'P': 2,}}
data.replace(classnumlize,inplace = True)
data['Source.Name'].value_counts()
data.head()


# In[8]:


# 把Label單一抓出來，並且將data列drop掉
Label = data['Source.Name']
data.drop( ['Source.Name'],axis = 1,inplace = True)
Label.head()


# In[9]:


#檢查feature
feature = pd.get_dummies(data)
feature.head()


# In[10]:


#標準化 
std=StandardScaler()
std.fit(feature)
feature_std=std.transform(feature)
#拆分
feature_train,feature_test,Label_train,Label_test = train_test_split(feature_std,Label,test_size=0.2, 
                     stratify=Label,random_state = 0)


# In[11]:


cov_mat = np.cov(feature_train.T)
eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)

print('\nEigenvalues \n%s' % eigen_vals)

tot = sum(eigen_vals)
var_exp = [(i / tot) for i in sorted(eigen_vals, reverse=True)]
cum_var_exp = np.cumsum(var_exp)


plt.bar(range(1, 33), var_exp, alpha=0.5, align='center',
        label='Individual explained variance')
plt.step(range(1, 33), cum_var_exp, where='mid',
         label='Cumulative explained variance')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal component index')
plt.legend(loc='best')
plt.tight_layout()
plt.show()


# In[12]:


pipe_lr = make_pipeline(LogisticRegression(penalty='l2', random_state=1,solver='lbfgs', max_iter=10000))

train_sizes, train_scores, test_scores =                learning_curve(estimator=pipe_lr,
                               X=feature_train,
                               y=Label_train,
                               train_sizes=np.linspace(0.1, 1.0, 10),
                               cv=15,
                               n_jobs=1)

train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

plt.plot(train_sizes, train_mean,
         color='blue', marker='o',
         markersize=5, label='Training accuracy')

plt.fill_between(train_sizes,
                 train_mean + train_std,
                 train_mean - train_std,
                 alpha=0.15, color='blue')

test_mean2 = sorted(test_mean)
plt.plot(train_sizes, test_mean2,
         color='green', linestyle='--',
         marker='s', markersize=5,
         label='Validation accuracy')

plt.fill_between(train_sizes,
                 test_mean2 + 0.25*test_std,
                 test_mean2 - 0.25*test_std,
                 alpha=0.15, color='green')

plt.grid()
plt.xlabel('Number of training examples')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.ylim([0.3, 1.03])
plt.tight_layout()
plt.show()


# In[13]:


#換另外一種命名方式 把它全部變成Value，將data拆分訓練集和測試集
X = feature.values
y = Label.values
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2, 
                    stratify=y,random_state = 0)
print(X_train.shape,X_test.shape,y_train.shape,y_test.shape)


# In[14]:


# K-fold CV for K = 3 (LogisticRegression)
kfold = StratifiedKFold(n_splits=3).split(X_train, y_train)
pipe_lr = make_pipeline(StandardScaler(),
            LogisticRegression(penalty='l2', C=10000,random_state=500, solver='lbfgs'))
scores = []
for k, (train, test) in enumerate(kfold):
    pipe_lr.fit(X_train[train], y_train[train])
    score = pipe_lr.score(X_train[test], y_train[test])
    scores.append(score)
    print('Fold: %2d, Class dist.: %s, Acc: %.3f' % (k+1,
          np.bincount(y_train[train]), score))
    
print('\nCV accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))

plt.plot(scores,
         color='blue', marker='o',
         markersize=5, label='Validation accuracy')
plt.grid()
plt.xlabel('Number of K')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.ylim([0.3, 1.03])
plt.tight_layout()
plt.show()


# In[15]:


# K-fold CV for K = 10 (LogisticRegression)
kfold = StratifiedKFold(n_splits=10).split(X_train, y_train)
pipe_lr = make_pipeline(StandardScaler(),
            LogisticRegression(penalty='l2', C=10000,random_state=50, solver='lbfgs'))
scores = []
for k, (train, test) in enumerate(kfold):
    pipe_lr.fit(X_train[train], y_train[train])
    score = pipe_lr.score(X_train[test], y_train[test])
    scores.append(score)
    print('Fold: %2d, Class dist.: %s, Acc: %.3f' % (k+1,
          np.bincount(y_train[train]), score))
    
print('\nCV accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))

plt.plot(scores,
         color='blue', marker='o',
         markersize=5, label='Validation accuracy')
plt.grid()
plt.xlabel('Number of K')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.ylim([0.3, 1.03])
plt.tight_layout()
plt.show()


# In[16]:


# K-fold CV for K = 3 (DecisionTree)
kfold = StratifiedKFold(n_splits=3).split(X_train, y_train)
pipe_lr = make_pipeline(DecisionTreeClassifier(max_depth=3, criterion='entropy',random_state=500))
scores = []
for k, (train, test) in enumerate(kfold):
    pipe_lr.fit(X_train[train], y_train[train])
    score = pipe_lr.score(X_train[test], y_train[test])
    scores.append(score)
    print('Fold: %2d, Class dist.: %s, Acc: %.3f' % (k+1,
          np.bincount(y_train[train]), score))
    
print('\nCV accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))

plt.plot(scores,
         color='blue', marker='o',
         markersize=5, label='Validation accuracy')
plt.grid()
plt.xlabel('Number of K')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.ylim([0.3, 1.03])
plt.tight_layout()
plt.show()


# In[17]:


# K-fold CV for K = 10 (DecisionTree)
kfold = StratifiedKFold(n_splits=10).split(X_train, y_train)
pipe_lr = make_pipeline(DecisionTreeClassifier(max_depth=3, criterion='entropy',random_state=1))
scores = []
for k, (train, test) in enumerate(kfold):
    pipe_lr.fit(X_train[train], y_train[train])
    score = pipe_lr.score(X_train[test], y_train[test])
    scores.append(score)
    print('Fold: %2d, Class dist.: %s, Acc: %.3f' % (k+1,
          np.bincount(y_train[train]), score))
    
print('\nCV accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))

plt.plot(scores,
         color='blue', marker='o',
         markersize=5, label='Validation accuracy')
plt.grid()
plt.xlabel('Number of K')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.ylim([0.3, 1.03])
plt.tight_layout()
plt.show()


# In[18]:


# K-fold CV for K = 10 (SVM-rbf)
kfold = StratifiedKFold(n_splits=10).split(X_train, y_train)
pipe_svc = make_pipeline(StandardScaler(),SVC(kernel='rbf',gamma=10, C=1,random_state=500))
scores = []
for k, (train, test) in enumerate(kfold):
    pipe_svc.fit(X_train[train], y_train[train])
    score = pipe_svc.score(X_train[test], y_train[test])
    scores.append(score)
    print('Fold: %2d, Class dist.: %s, Acc: %.3f' % (k+1,
          np.bincount(y_train[train]), score))
    
print('\nCV accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))

plt.plot(scores,
         color='blue', marker='o',
         markersize=5, label='Validation accuracy')
plt.grid()
plt.xlabel('Number of K')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.ylim([0.3, 1.03])
plt.tight_layout()
plt.show()


# In[19]:


# K-fold CV for K = 3 (SVM-rbf)
kfold = StratifiedKFold(n_splits=3).split(X_train, y_train)
pipe_svc = make_pipeline(StandardScaler(),SVC(kernel='rbf',gamma=10, C=1,random_state=500))
scores = []
for k, (train, test) in enumerate(kfold):
    pipe_svc.fit(X_train[train], y_train[train])
    score = pipe_svc.score(X_train[test], y_train[test])
    scores.append(score)
    print('Fold: %2d, Class dist.: %s, Acc: %.3f' % (k+1,
          np.bincount(y_train[train]), score))
    
print('\nCV accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))

plt.plot(scores,
         color='blue', marker='o',
         markersize=5, label='Validation accuracy')
plt.grid()
plt.xlabel('Number of K')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.ylim([0.3, 1.03])
plt.tight_layout()
plt.show()


# In[20]:


# K-fold CV for K = 10 (SVM-sigmoid)
kfold = StratifiedKFold(n_splits=10).split(X_train, y_train)
pipe_svc = make_pipeline(StandardScaler(),SVC(kernel='sigmoid',gamma=0.001,C=1000,random_state=0))
scores = []
for k, (train, test) in enumerate(kfold):
    pipe_svc.fit(X_train[train], y_train[train])
    score = pipe_svc.score(X_train[test], y_train[test])
    scores.append(score)
    print('Fold: %2d, Class dist.: %s, Acc: %.3f' % (k+1,
          np.bincount(y_train[train]), score))
    
print('\nCV accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))

plt.plot(scores,
         color='blue', marker='o',
         markersize=5, label='Validation accuracy')
plt.grid()
plt.xlabel('Number of K')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.ylim([0.3, 1.03])
plt.tight_layout()
plt.show()


# In[21]:


# K-fold CV for K = 3 (SVM-sigmoid)
kfold = StratifiedKFold(n_splits=3).split(X_train, y_train)
pipe_svc = make_pipeline(StandardScaler(),SVC(kernel='sigmoid',gamma=0.001,C=1000,random_state=20))
scores = []
for k, (train, test) in enumerate(kfold):
    pipe_svc.fit(X_train[train], y_train[train])
    score = pipe_svc.score(X_train[test], y_train[test])
    scores.append(score)
    print('Fold: %2d, Class dist.: %s, Acc: %.3f' % (k+1,
          np.bincount(y_train[train]), score))
    
print('\nCV accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))

plt.plot(scores,
         color='blue', marker='o',
         markersize=5, label='Validation accuracy')
plt.grid()
plt.xlabel('Number of K')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.ylim([0.3, 1.03])
plt.tight_layout()
plt.show()


# In[22]:


# K-fold CV for K = 3 (SVM-Poly)
kfold = StratifiedKFold(n_splits=3).split(X_train, y_train)
pipe_svc = make_pipeline(StandardScaler(),SVC(kernel='poly',C=0.001,gamma=0.001,degree=2,random_state=0))
scores = []
for k, (train, test) in enumerate(kfold):
    pipe_svc.fit(X_train[train], y_train[train])
    score = pipe_svc.score(X_train[test], y_train[test])
    scores.append(score)
    print('Fold: %2d, Class dist.: %s, Acc: %.3f' % (k+1,
          np.bincount(y_train[train]), score))
    
print('\nCV accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))

plt.plot(scores,
         color='blue', marker='o',
         markersize=5, label='Validation accuracy')
plt.grid()
plt.xlabel('Number of K')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.ylim([0.3, 1.03])
plt.tight_layout()
plt.show()


# In[23]:


# K-fold CV for K = 10 (SVM-Poly)
kfold = StratifiedKFold(n_splits=10).split(X_train, y_train)
pipe_svc = make_pipeline(StandardScaler(),SVC(kernel='poly',C=0.001,gamma=0.001,degree=2,random_state=0))
scores = []
for k, (train, test) in enumerate(kfold):
    pipe_svc.fit(X_train[train], y_train[train])
    score = pipe_svc.score(X_train[test], y_train[test])
    scores.append(score)
    print('Fold: %2d, Class dist.: %s, Acc: %.3f' % (k+1,
          np.bincount(y_train[train]), score))
    
print('\nCV accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))

plt.plot(scores,
         color='blue', marker='o',
         markersize=5, label='Validation accuracy')
plt.grid()
plt.xlabel('Number of K')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.ylim([0.3, 1.03])
plt.tight_layout()
plt.show()


# In[24]:


# K-fold CV for K = 10 (SVM-linear)
kfold = StratifiedKFold(n_splits=10).split(X_train, y_train)
pipe_svc = make_pipeline(StandardScaler(),SVC(kernel='linear',C=10,random_state=0))
scores = []
for k, (train, test) in enumerate(kfold):
    pipe_svc.fit(X_train[train], y_train[train])
    score = pipe_svc.score(X_train[test], y_train[test])
    scores.append(score)
    print('Fold: %2d, Class dist.: %s, Acc: %.3f' % (k+1,
          np.bincount(y_train[train]), score))
    
print('\nCV accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))

plt.plot(scores,
         color='blue', marker='o',
         markersize=5, label='Validation accuracy')
plt.grid()
plt.xlabel('Number of K')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.ylim([0.3, 1.03])
plt.tight_layout()
plt.show()


# In[25]:


# K-fold CV for K = 3 (SVM-linear)
kfold = StratifiedKFold(n_splits=3).split(X_train, y_train)
pipe_svc = make_pipeline(StandardScaler(),SVC(kernel='linear',C=10,random_state=0))
scores = []
for k, (train, test) in enumerate(kfold):
    pipe_svc.fit(X_train[train], y_train[train])
    score = pipe_svc.score(X_train[test], y_train[test])
    scores.append(score)
    print('Fold: %2d, Class dist.: %s, Acc: %.3f' % (k+1,
          np.bincount(y_train[train]), score))
    
print('\nCV accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))

plt.plot(scores,
         color='blue', marker='o',
         markersize=5, label='Validation accuracy')
plt.grid()
plt.xlabel('Number of K')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.ylim([0.3, 1.03])
plt.tight_layout()
plt.show()


# In[26]:


# CV調節 LR   (penalty='l2',solver='lbfgs')

pipe_lr = make_pipeline(StandardScaler(),LogisticRegression(penalty='l2', random_state=1,solver='lbfgs', max_iter=1000))

param_range = [0.0001 ,0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]
train_scores, test_scores = validation_curve(estimator=pipe_lr, X=X_train, y=y_train, 
                param_name='logisticregression__C', param_range=param_range,cv=10)

train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

plt.plot(param_range, train_mean, 
         color='blue', marker='o', 
         markersize=5, label='Training accuracy')

plt.fill_between(param_range, train_mean + train_std,
                 train_mean - train_std, alpha=0.15,
                 color='blue')

plt.plot(param_range, test_mean, 
         color='green', linestyle='--', 
         marker='s', markersize=5, 
         label='Validation accuracy')

plt.fill_between(param_range, 
                 test_mean ,
                 test_mean , 
                 alpha=0.15, color='green')

plt.grid()
plt.xscale('log')
plt.legend(loc='lower right')
plt.xlabel('Parameter C')
plt.ylabel('Accuracy')
plt.ylim([0.3, 1.03])
plt.tight_layout()
plt.show()


# In[27]:


# CV調節 LR   (penalty='l2',solver='newton-cg')

pipe_lr = make_pipeline(StandardScaler(),LogisticRegression(penalty='l2', random_state=1,solver='newton-cg', max_iter=1000))

param_range = [0.0001 ,0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]
train_scores, test_scores = validation_curve(estimator=pipe_lr, X=X_train, y=y_train, 
                param_name='logisticregression__C', param_range=param_range,cv=10)

train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

plt.plot(param_range, train_mean, color='blue', marker='o', markersize=5, label='Training accuracy')
plt.fill_between(param_range, train_mean + train_std,train_mean - train_std, alpha=0.15,color='blue')
plt.plot(param_range, test_mean,color='green', linestyle='--', marker='s', markersize=5,label='Validation accuracy')
plt.fill_between(param_range, test_mean ,test_mean , alpha=0.15, color='green')

plt.grid()
plt.xscale('log')
plt.legend(loc='lower right')
plt.xlabel('Parameter C')
plt.ylabel('Accuracy')
plt.ylim([0.3, 1.03])
plt.tight_layout()
plt.show()


# In[28]:


# CV調節 LR   (penalty='l2',solver='sag')

pipe_lr = make_pipeline(StandardScaler(),LogisticRegression(penalty='l2', random_state=1,solver='sag', max_iter=1000))

param_range = [0.0001 ,0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]
train_scores, test_scores = validation_curve(estimator=pipe_lr, X=X_train, y=y_train, 
                param_name='logisticregression__C', param_range=param_range,cv=10)

train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

plt.plot(param_range, train_mean, color='blue', marker='o', markersize=5, label='Training accuracy')
plt.fill_between(param_range, train_mean + train_std,train_mean - train_std, alpha=0.15,color='blue')
plt.plot(param_range, test_mean,color='green', linestyle='--', marker='s', markersize=5,label='Validation accuracy')
plt.fill_between(param_range, test_mean ,test_mean , alpha=0.15, color='green')

plt.grid()
plt.xscale('log')
plt.legend(loc='lower right')
plt.xlabel('Parameter C')
plt.ylabel('Accuracy')
plt.ylim([0.3, 1.03])
plt.tight_layout()
plt.show()


# In[29]:


# GS調節DT
regressor = DecisionTreeClassifier(random_state=0)
parameters = {'max_depth': range(1, 10)}
scoring_fnc = make_scorer(accuracy_score)
kfold = KFold(n_splits=10)

grid = GridSearchCV(regressor, parameters, scoring_fnc, cv=kfold)
grid = grid.fit(X_train, y_train)
reg = grid.best_estimator_

print('best score: %f'%grid.best_score_)
print('best parameters:')
for key in parameters.keys():
    print('%s: %d'%(key, reg.get_params()[key]))

print('test score: %f'%reg.score(X_test, y_test))


# In[30]:


# GS調節Poly
pipe_svc = make_pipeline(StandardScaler(),SVC(random_state=1))
scorer = make_scorer(score_func=precision_score, 
                         pos_label=1, 
                         greater_is_better=True, 
                         average='micro')

c_gamma_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0] 
degree_range = [2,3,4]
param_grid = [{'svc__C': c_gamma_range, 
               'svc__gamma': c_gamma_range,
               'svc__degree' : degree_range,
               'svc__kernel': ['poly']}]

gs = GridSearchCV(estimator=pipe_svc, 
                  param_grid=param_grid, 
                  scoring='accuracy', 
                  refit=True,
                  cv=10,
                  n_jobs=-1)
gs = gs.fit(X_train, y_train)

print("Best parameters set found on development set:")
print()
print(gs.best_params_)
print()
print("Grid scores on development set:")
print()
means = gs.cv_results_['mean_test_score']
stds = gs.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, gs.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r"
            % (mean, std * 2, params))
print()    
print('-----------------------------------------------')     
print("Classification report:")
print("The model is trained on the full development set.")
print("The scores are computed on the full evaluation set.")
print()
print("Best train accuracy from gridsearch:")
print(gs.best_score_)
print("Best parameters from gridsearch:")
print(gs.best_params_)
clf = gs.best_estimator_
print('Test accuracy: %.3f' % clf.score(X_test, y_test))
print('-----------------------------------------------') 


# In[31]:


# GS調節SVM_RBF
pipe_svc = make_pipeline(StandardScaler(),SVC(random_state=1))
scorer = make_scorer(score_func=precision_score, 
                         pos_label=1, 
                         greater_is_better=True, 
                         average='micro')

c_gamma_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0] 
param_grid = [{'svc__C': c_gamma_range,
               'svc__gamma': c_gamma_range,
               'svc__kernel': ['rbf']}]

gs = GridSearchCV(estimator=pipe_svc, 
                  param_grid=param_grid, 
                  scoring='accuracy', 
                  refit=True,
                  cv=10,
                  n_jobs=-1)
gs = gs.fit(X_train, y_train)

print("Best parameters set found on development set:")
print()
print(gs.best_params_)
print()
print("Grid scores on development set:")
print()
means = gs.cv_results_['mean_test_score']
stds = gs.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, gs.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r"
            % (mean, std * 2, params))
print()    
print('-----------------------------------------------')     
print("Classification report:")
print("The model is trained on the full development set.")
print("The scores are computed on the full evaluation set.")
print()
print("Best train accuracy from gridsearch:")
print(gs.best_score_)
print("Best parameters from gridsearch:")
print(gs.best_params_)
clf = gs.best_estimator_
print('Test accuracy: %.3f' % clf.score(X_test, y_test))
print('-----------------------------------------------') 


# In[32]:


# GS調節SVM-Linear
pipe_svc = make_pipeline(StandardScaler(),SVC(random_state=10000))
scorer = make_scorer(score_func=precision_score, 
                         pos_label=1, 
                         greater_is_better=True, 
                         average='micro')

c_gamma_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0] 
param_grid = [{'svc__C': c_gamma_range,
               'svc__kernel': ['linear']}]

gs = GridSearchCV(estimator=pipe_svc, 
                  param_grid=param_grid, 
                  scoring='accuracy', 
                  refit=True,
                  cv=10,
                  n_jobs=-1)
gs = gs.fit(X_train, y_train)

print("Best parameters set found on development set:")
print()
print(gs.best_params_)
print()
print("Grid scores on development set:")
print()
means = gs.cv_results_['mean_test_score']
stds = gs.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, gs.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r"
            % (mean, std * 2, params))
print()    
print('-----------------------------------------------')     
print("Classification report:")
print("The model is trained on the full development set.")
print("The scores are computed on the full evaluation set.")
print()
print("Best train accuracy from gridsearch:")
print(gs.best_score_)
print("Best parameters from gridsearch:")
print(gs.best_params_)
clf = gs.best_estimator_
print('Test accuracy: %.3f' % clf.score(X_test, y_test))
print('-----------------------------------------------') 


# In[33]:


# GS調節SVM_sigmoid
pipe_svc = make_pipeline(StandardScaler(),SVC(random_state=2))
scorer = make_scorer(score_func=precision_score, 
                         pos_label=1, 
                         greater_is_better=True, 
                         average='micro')

c_gamma_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0] 
param_grid = [{'svc__C': c_gamma_range,
               'svc__gamma': c_gamma_range,
               'svc__kernel': ['sigmoid']}]

gs = GridSearchCV(estimator=pipe_svc, 
                  param_grid=param_grid, 
                  scoring='accuracy', 
                  refit=True,
                  cv=10,
                  n_jobs=-1)
gs = gs.fit(X_train, y_train)

print("Best parameters set found on development set:")
print()
print(gs.best_params_)
print()
print("Grid scores on development set:")
print()
means = gs.cv_results_['mean_test_score']
stds = gs.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, gs.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r"
            % (mean, std * 2, params))
print()    
print('-----------------------------------------------')     
print("Classification report:")
print("The model is trained on the full development set.")
print("The scores are computed on the full evaluation set.")
print()
print("Best train accuracy from gridsearch:")
print(gs.best_score_)
print("Best parameters from gridsearch:")
print(gs.best_params_)
clf = gs.best_estimator_
print('Test accuracy: %.3f' % clf.score(X_test, y_test))
print('-----------------------------------------------') 


# In[34]:


clf1 = LogisticRegression(penalty='l2', C=0.1,solver='lbfgs',random_state=500)

clf2 = DecisionTreeClassifier(max_depth=3, criterion='entropy',random_state=500)

clf3 = SVC(kernel='rbf',gamma=10, C=1,random_state=500)

clf4 = SVC(kernel='sigmoid',gamma=0.001,C=1000,random_state=500)

clf5 = SVC(kernel='poly',C=0.0001,gamma=0.0001,degree=2,random_state=500)

clf6 = SVC(kernel='linear',C=10,random_state=500)


pipe1 = Pipeline([['sc', StandardScaler()],
                  ['clf', clf1]])
pipe3 = Pipeline([['sc', StandardScaler()],
                  ['clf', clf3]])
pipe4 = Pipeline([['sc', StandardScaler()],
                  ['clf', clf4]])
pipe5 = Pipeline([['sc', StandardScaler()],
                  ['clf', clf5]])
pipe6 = Pipeline([['sc', StandardScaler()],
                  ['clf', clf6]])

eclf1 = VotingClassifier(estimators=[('lr', pipe1), ('dt', clf2), ('svm_rbf', pipe3), ('svm_sig', pipe4),
                                     ('svm_poly', pipe5), ('svm_linear', pipe6)], voting='hard')
eclf2 = VotingClassifier(estimators=[('lr', pipe1), ('dt', clf2), ('svm_rbf', pipe3), ('svm_sig', pipe4),
                                     ('svm_poly', pipe5), ('svm_linear', pipe6)], voting='hard', 
                         weights=[1,3,4,1,1,2], flatten_transform=True)

clf_labels = ['LogisticRegression model','DecisionTree model','svm_RBF model','svm_Sigmoid model','svm_Poly model','svm_linear model']


pipe1.fit(X_train, y_train)
print('---LogisticRegression model---')
print('Training accuracy:', pipe1.score(X_train, y_train))
print('Test accuracy:', pipe1.score(X_test, y_test))

clf2.fit(X_train, y_train)
print('---DecisionTree model---')
print('Training accuracy:', clf2.score(X_train, y_train))
print('Test accuracy:', clf2.score(X_test, y_test))


pipe3.fit(X_train, y_train)
print('---svm_RBF model---')
print('Training accuracy:', pipe3.score(X_train, y_train))
print('Test accuracy:', pipe3.score(X_test, y_test))

pipe4.fit(X_train, y_train)
print('---svm_Sigmoid model---')
print('Training accuracy:', pipe4.score(X_train, y_train))
print('Test accuracy:', pipe4.score(X_test, y_test))

pipe5.fit(X_train, y_train)
print('---svm_Poly model---')
print('Training accuracy:', pipe5.score(X_train, y_train))
print('Test accuracy:', pipe5.score(X_test, y_test))

pipe6.fit(X_train, y_train)
print('---svm_linear model---')
print('Training accuracy:', pipe6.score(X_train, y_train))
print('Test accuracy:', pipe6.score(X_test, y_test))


eclf1.fit(X, y)
print('---Hard Majority Vote---')
print('Training accuracy:', eclf1.score(X_train, y_train))
print('Test accuracy:', eclf1.score(X_test, y_test))

eclf2.fit(X, y)
print('---Soft Majority Vote---')
print('Training accuracy:', eclf2.score(X_train, y_train))
print('Test accuracy:', eclf2.score(X_test, y_test))


# In[35]:


kfold = StratifiedKFold(n_splits=10).split(X_train, y_train)

clf_labels = ['1.Logistic regression', '2.Decision tree', '3.svm_RBF', '4.svm_Sigmoid', 
              '5.svm_Polynomial kernal', '6.svm_Linear','7.Bagging ', '8.Majority weighted Vote']
scores = []

print('K-fold cross validation:\n')
for clf, label in zip([pipe1, eclf2, pipe3, pipe4, pipe5, pipe6, eclf1, clf2], clf_labels):
    scores = cross_val_score(estimator=clf,X=X_train,y=y_train,cv=10)
    print("cross validation score: %0.2f (+/- %0.2f) [%s]"% (scores.mean(), scores.std(), label))


# In[36]:


joblib.dump(eclf1, "Tool_Life_Classification_Model/0805Mjv_model.dat")
joblib.dump(eclf2, "Tool_Life_Classification_Model/0805rMjv_model.dat")


# In[88]:


joblib.dump(clf2, "Tool_Life_Classification_Model/0716DT_model.dat")
joblib.dump(pipe1, "Tool_Life_Classification_Model/0716LRM_model.dat")
joblib.dump(pipe3, "Tool_Life_Classification_Model/0716RBF_model.dat")
joblib.dump(pipe4, "Tool_Life_Classification_Model/0716Sig_model.dat")
joblib.dump(pipe5, "Tool_Life_Classification_Model/0716Poly_model.dat")
joblib.dump(pipe6, "Tool_Life_Classification_Model/0716Line_model.dat")


# In[37]:


from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.base import clone
from sklearn.pipeline import _name_estimators
import numpy as np
import operator


class MajorityVoteClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, classifiers, vote='classlabel', weights=None):

        self.classifiers = classifiers
        self.named_classifiers = {key: value for key, value
                                  in _name_estimators(classifiers)}
        self.vote = vote
        self.weights = weights

    def fit(self, X, y):

        if self.vote not in ('probability', 'classlabel'):
            raise ValueError("vote must be 'probability' or 'classlabel'"
                             "; got (vote=%r)"
                             % self.vote)

        if self.weights and len(self.weights) != len(self.classifiers):
            raise ValueError('Number of classifiers and weights must be equal'
                             '; got %d weights, %d classifiers'
                             % (len(self.weights), len(self.classifiers)))

        # Use LabelEncoder to ensure class labels start with 0, which
        # is important for np.argmax call in self.predict
        self.lablenc_ = LabelEncoder()
        self.lablenc_.fit(y)
        self.classes_ = self.lablenc_.classes_
        self.classifiers_ = []
        for clf in self.classifiers:
            fitted_clf = clone(clf).fit(X, self.lablenc_.transform(y))
            self.classifiers_.append(fitted_clf)
        return self

    def predict(self, X):
        if self.vote == 'probability':
            maj_vote = np.argmax(self.predict_proba(X), axis=1)
        else: 
            predictions = np.asarray([clf.predict(X)
                                      for clf in self.classifiers_]).T

            maj_vote = np.apply_along_axis(
                                      lambda x:
                                      np.argmax(np.bincount(x,
                                                weights=self.weights)),
                                      axis=1,
                                      arr=predictions)
        maj_vote = self.lablenc_.inverse_transform(maj_vote)
        return maj_vote

    def predict_proba(self, X):
        probas = np.asarray([clf.predict_proba(X)
                             for clf in self.classifiers_])
        avg_proba = np.average(probas, axis=0, weights=self.weights)
        return avg_proba

    def get_params(self, deep=True):
        if not deep:
            return super(MajorityVoteClassifier, self).get_params(deep=False)
        else:
            out = self.named_classifiers.copy()
            for name, step in self.named_classifiers.items():
                for key, value in step.get_params(deep=True).items():
                    out['%s__%s' % (name, key)] = value
            return out


# In[56]:


mv_clf = MajorityVoteClassifier(classifiers=[pipe1, clf2, pipe3, pipe4, pipe5, pipe6])
clf_labels += ['Majority voting']
all_clf = [pipe1, clf2, pipe3, pipe4, pipe5, pipe6, mv_clf]

for clf, label in zip(all_clf, clf_labels):
    scores = cross_val_score(estimator=clf,
                             X=X_train,
                             y=y_train,
                             cv=10)
    print("accuracy: %0.2f (+/- %0.2f) [%s]"
          % (scores.mean(), scores.std(), label))


# In[57]:


mv_clf.get_params()


# In[38]:


from sklearn.ensemble import AdaBoostClassifier

tree = DecisionTreeClassifier(criterion='entropy', 
                              max_depth=1,
                              random_state=10)

ada = AdaBoostClassifier(base_estimator=tree,
                         n_estimators=1000, 
                         learning_rate=0.1,
                         random_state=10)

tree = tree.fit(X_train, y_train)
y_train_pred = tree.predict(X_train)
y_test_pred = tree.predict(X_test)

tree_train = accuracy_score(y_train, y_train_pred)
tree_test = accuracy_score(y_test, y_test_pred)
print('Decision tree train/test accuracy %.14f/%.14f'
      % (tree_train, tree_test))

ada = ada.fit(X_train, y_train)
y_train_pred = ada.predict(X_train)
y_test_pred = ada.predict(X_test)

ada_train = accuracy_score(y_train, y_train_pred) 
ada_test = accuracy_score(y_test, y_test_pred) 
print('AdaBoost train/test accuracy %.14f/%.14f'
      % (ada_train, ada_test))


# In[39]:


joblib.dump(ada, "Tool_Life_Classification_Model/0807adaboost_model.dat")

