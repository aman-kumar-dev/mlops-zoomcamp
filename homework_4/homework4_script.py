#!/usr/bin/env python
# coding: utf-8

# In[1]:


# get_ipython().system('pip freeze | grep scikit-learn')


# In[ ]:


import pickle
import pandas as pd
import numpy as np
import sys


# In[ ]:


with open('model.bin', 'rb') as f_in:
    dv, lr = pickle.load(f_in)


# In[ ]:


categorical = ['PUlocationID', 'DOlocationID']

def read_data(filename):
    df = pd.read_parquet(filename)
    
    df['duration'] = df.dropOff_datetime - df.pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df


# In[ ]:

YYYY, MM = sys.argv[1], sys.argv[2]
link = f"https://nyc-tlc.s3.amazonaws.com/trip+data/fhv_tripdata_{YYYY}-{MM}.parquet"
df = read_data(link)

# In[ ]:


dicts = df[categorical].to_dict(orient='records')
X_val = dv.transform(dicts)
y_pred = lr.predict(X_val)
print(f"Mean pred duration is {np.mean(y_pred)}")

