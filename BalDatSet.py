#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pickle
import pandas as pd


# In[2]:


with open('label.pkl', 'rb') as f:
      dt = pickle.load(f)


# In[3]:


with open('flabel.pkl', 'rb') as f:
      label = pickle.load(f)


# In[4]:


pd.set_option('display.max_colwidth', 1000000)


# In[ ]:


# Creation of a balanced dataset


# In[ ]:


header_list = ['Tweet','Class']
dt = dt.reindex(columns = header_list)


# In[8]:


for i in range(0,len(label)) :
    if label[i] == 0 :
        dt.set_value(i, 'Class', 0) 
    else :
        if label[i] == 1 :
            dt.set_value(i, 'Class', 1)


# In[ ]:


with open('datf.pkl', 'wb') as f:
     pickle.dump(dt, f)

