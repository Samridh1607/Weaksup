#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pickle
with open('label.pkl', 'rb') as f:
      dt = pickle.load(f)


# In[5]:


import pandas as pd
pd.set_option('display.max_colwidth', 1000000)


# In[6]:


# Module 2 (Label this data)
# Use snorkel
from snorkel.labeling import labeling_function
from snorkel.preprocess import preprocessor
import re
from textblob import TextBlob  


# In[7]:


# Create labelling function based on words in the corpus
from snorkel.labeling import LabelingFunction
PWORDS = r"""\b(notmyimpeachment|do not impeach|meaningless impeachment|unfair|fair trial|not enough evidence|miss conduct|delay trial|fake impeachment)"""
def pos(sample):
        return 1 if re.search(PWORDS, str(sample)) else -1

NWORDS = r"\b(hell yeah|bribery|not happy|less moral|impeach trump|impeach our president)"
def neg(sample):
    return 0 if re.search(NWORDS, str(sample)) else -1


# In[8]:


positive = LabelingFunction(f"positive", f=pos)


# In[9]:


negative = LabelingFunction(f"negative", f=neg)


# In[10]:


df_train = dt[0:450000]


# In[11]:


# Create the labeling functions using the textblob sentiment analyzer
@preprocessor(memoize=True)
def textblob_polarity(x):
    scores = TextBlob(x.Tweet)
    x.polarity = scores.polarity
    return x


# Label high polarity tweets as positive.
@labeling_function(pre=[textblob_polarity])
def polarity_positive(x):
    return 1 if x.polarity > 0.2 else -1


# Label low polarity tweets as negative.
@labeling_function(pre=[textblob_polarity])
def polarity_negative(x):
    return 0 if x.polarity < -0.25 else -1


# Similar to polarity_negative, but with higher coverage and lower precision.
@labeling_function(pre=[textblob_polarity])
def polarity_negative_2(x):
    return 0 if x.polarity <= 0.2 else -1


# In[12]:


from snorkel.labeling import PandasLFApplier
wlfs = [positive,negative]
blob_lfs = [polarity_positive, polarity_negative,polarity_negative_2]
lfs = wlfs + blob_lfs 
applier = PandasLFApplier(lfs)
L_train = applier.apply(df_train)


# In[19]:


L_train
# Analyse the coverage and accuracy
from snorkel.labeling import LFAnalysis
LFAnalysis(L=L_train, lfs=lfs).lf_summary()


# In[12]:


print(f"Training set coverage: {100 * LFAnalysis(L_train).label_coverage(): 0.001f}%")


# In[15]:


from snorkel.labeling import LabelModel

# Train LabelModel.
label_model = LabelModel(cardinality=2, verbose=True)
label_model.fit(L_train, n_epochs=150, seed=125, log_freq=30, l2=0.1, lr=0.01)


# In[16]:


label = label_model.predict(L_train)


# In[18]:


len(label)


# In[26]:





# In[ ]:


with open('flabel.pkl', 'wb') as f:
     pickle.dump(label, f)

