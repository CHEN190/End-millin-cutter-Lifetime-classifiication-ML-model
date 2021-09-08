#!/usr/bin/env python
# coding: utf-8

# In[2]:


import sklearn.externals
import joblib
import pandas as pd # 引用套件並縮寫為pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore") # 把warning關掉

from sklearn import datasets
import numpy as np
from IPython.display import Image

get_ipython().run_line_magic('matplotlib', 'inline')


# In[40]:


predata = pd.read_csv(r'C:\Users\user\Tool_Life_Classification_Model\‪TC_4_JSK_D12_190630_Batch1_B16.csv')
predata.head()


# In[4]:


predata.drop( ['Source.name'],axis = 1,inplace = True)
predata.head()


# In[6]:


predata = pd.concat([data, predata], sort=False)
predata.head()


# In[7]:


#標準化 
from sklearn.preprocessing import StandardScaler 
std=StandardScaler()
std.fit(predata)
predata_std=std.transform(predata)
print(predata_std[0:5,:])


# In[8]:


testdata = predata_std[0,:]
print(testdata)


# In[3]:


Model_LRM = joblib.load("0716LRM_model.dat")
Model_DT = joblib.load("0716DT_model.dat")
Model_RBF = joblib.load("0716RBF_model.dat")
Model_Line = joblib.load("0716Line_model.dat")
Model_Poly = joblib.load("0716Poly_model.dat")
Model_Sig = joblib.load("0716Sig_model.dat")
Model_Mjv = joblib.load("0805Mjv_model.dat")
Model_rMjv = joblib.load("0805rMjv_model.dat")
Model_Adb = joblib.load("0807adaboost_model.dat")


# In[4]:


data  = pd.read_csv(r'C:\Users\user\Tool_Life_Classification_Model\‪TC_4_JSK_D12_190630_Batch1_B16.csv')
data2 = pd.read_csv(r'C:\Users\user\Tool_Life_Classification_Model\‪TC_4_JSK_D12_190630_Batch1_RR10.csv')
data3 = pd.read_csv(r'C:\Users\user\Tool_Life_Classification_Model\‪TC_4_JSK_D12_190630_Batch1_PP14.csv')
print(data3)


# In[77]:


y_pred = Model_DT.predict(data)
print(y_pred)
y_pred = Model_LRM.predict(data)
print(y_pred)
y_pred = Model_Line.predict(data)
print(y_pred)
y_pred = Model_Poly.predict(data)
print(y_pred)
y_pred = Model_Sig.predict(data)
print(y_pred)
y_pred = Model_RBF.predict(data)
print(y_pred)
# [0]為報廢刀刀具編碼 B，[1]為粗加工刀刀具編碼 R，[2]為精加工刀編碼 P


# In[78]:


y_pred = Model_DT.predict(data2)
print(y_pred)
y_pred = Model_LRM.predict(data2)
print(y_pred)
y_pred = Model_Line.predict(data2)
print(y_pred)
y_pred = Model_Poly.predict(data2)
print(y_pred)
y_pred = Model_Sig.predict(data2)
print(y_pred)
y_pred = Model_RBF.predict(data2)
print(y_pred)
# [0]為報廢刀刀具編碼 B，[1]為粗加工刀刀具編碼 R，[2]為精加工刀編碼 P


# In[79]:


y_pred = Model_DT.predict(data3)
print(y_pred)
y_pred = Model_LRM.predict(data3)
print(y_pred)
y_pred = Model_Line.predict(data3)
print(y_pred)
y_pred = Model_Poly.predict(data3)
print(y_pred)
y_pred = Model_Sig.predict(data3)
print(y_pred)
y_pred = Model_RBF.predict(data3)
print(y_pred)
# [0]為報廢刀刀具編碼 B，[1]為粗加工刀刀具編碼 R，[2]為精加工刀編碼 P


# In[55]:


y_pred = Model_DT.predict(data)
print(y_pred)
# [0]為報廢刀刀具編碼 B，[1]為粗加工刀刀具編碼 R，[2]為精加工刀編碼 P


# In[63]:


y_pred = Model_LRM.predict(data)
print(y_pred)
# [0]為報廢刀刀具編碼 B，[1]為粗加工刀刀具編碼 R，[2]為精加工刀編碼 P


# In[64]:


y_pred = Model_Line.predict(data)
print(y_pred)
# [0]為報廢刀刀具編碼 B，[1]為粗加工刀刀具編碼 R，[2]為精加工刀編碼 P


# In[65]:


y_pred = Model_Poly.predict(data)
print(y_pred)
# [0]為報廢刀刀具編碼 B，[1]為粗加工刀刀具編碼 R，[2]為精加工刀編碼 P


# In[66]:


y_pred = Model_Sig.predict(data)
print(y_pred)
# [0]為報廢刀刀具編碼 B，[1]為粗加工刀刀具編碼 R，[2]為精加工刀編碼 P


# In[67]:


y_pred = Model_RBF.predict(data)
print(y_pred)
# [0]為報廢刀刀具編碼 B，[1]為粗加工刀刀具編碼 R，[2]為精加工刀編碼 P


# In[81]:


y_pred = Model_Mjv.predict(data)
print(y_pred)
# [0]為報廢刀刀具編碼 B，[1]為粗加工刀刀具編碼 R，[2]為精加工刀編碼 P


# In[83]:


y_pred = Model_rMjv.predict(data)
print(y_pred)
# [0]為報廢刀刀具編碼 B，[1]為粗加工刀刀具編碼 R，[2]為精加工刀編碼 P


# In[113]:


y_pred = Model_Adb.predict(data)
print(y_pred)
# [0]為報廢刀刀具編碼 B，[1]為粗加工刀刀具編碼 R，[2]為精加工刀編碼 P


# In[ ]:




