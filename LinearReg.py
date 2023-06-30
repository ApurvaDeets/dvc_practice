#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import all the libraries


# In[2]:


import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report,accuracy_score,precision_score,recall_score
import pickle
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")
from imblearn.over_sampling import RandomOverSampler


# In[3]:


#import data
df=pd.read_csv("data1.csv")
print(df.shape)
print(df.columns.tolist())
df.head()
#to drop few columns which are not required in model building



# In[4]:


df1=df.copy() #to make a copy of the dataset
print(df1.shape)
df1.dropna(inplace=True)
print(df1.shape)
df1.head()
X=df1.drop(['Actual'],axis=1)
y=df1['Actual']
# y=df1["Actual"]
# df1.pop("Alloy")
# df1.pop("Alloy_encoded")
# df1.pop("Diameter_encoded")
# df1.pop("Id")
# df1.pop("start_timestamp")
# df1.pop("end_timestamp")



# In[5]:


X.head()


# In[6]:


y.value_counts()


# In[7]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)
from imblearn.over_sampling import RandomOverSampler


# In[8]:


ros=RandomOverSampler(sampling_strategy=0.7,random_state=42)


# In[9]:


X_train_res,y_train_res=ros.fit_resample(X_train,y_train)


# In[10]:


print(X_train_res.shape, y_train_res.shape)


# In[11]:


pd.DataFrame(y_train_res).value_counts()


# In[12]:


X_train_res.pop("Alloy_encoded")
X_train_res.pop("Id")
X_train_res.pop("start_timestamp")
X_train_res.pop("end_timestamp")


# In[13]:


X_test.pop("Alloy_encoded")
X_test.pop("Id")
X_test.pop("start_timestamp")
X_test.pop("end_timestamp")
X_test.pop("Alloy")
X_test


# In[14]:


X_train_res.pop("Alloy")
X_train_res


# In[15]:


X_train_res.isna().sum()


# In[16]:


get_ipython().run_cell_magic('time', '', 'model = LogisticRegression(random_state=42)\nlr=model.fit(X_train_res,y_train_res)\ny_pred_train = lr.predict(X_train_res)\nscore = accuracy_score(y_train_res, y_pred_train)\nprint("Training accuracy:", score)\nprint("*"*20)\nclf_report = classification_report(y_train_res, y_pred_train)\nprint("Classification Report :\\n",clf_report)\nprint("*"*60)\n\ny_pred_test = lr.predict(X_test)\nscore1 = accuracy_score(y_test, y_pred_test)\nprint("Testing accuracy:", score1)\nprint("*"*20)\nclf_report1 = classification_report(y_test, y_pred_test)\nprint("Classification Report :\\n",clf_report1)\n')

