#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


nfl_data = pd.read_csv("NFLPlaybyPlay.csv")


# In[3]:


nfl_data.head()


# In[4]:


missing_vals = nfl_data.isnull().sum()
missing_vals[0:20]


# In[5]:


total_cells = np.product(nfl_data.shape)
total_cells


# In[6]:


nfl_data.shape


# In[7]:


total_missing = missing_vals.sum()
total_missing


# In[8]:


449371*255


# In[9]:


percent_missing = (total_missing / total_cells) * 100 
percent_missing


# In[10]:


nfl_data.dropna()


# In[11]:


cols_with_NaN_dropped = nfl_data.dropna(axis=1)
cols_with_NaN_dropped.head()


# In[12]:


cols_with_NaN_dropped.columns.size


# In[13]:


cols_with_NaN_dropped.columns


# In[14]:


nfl_data.columns.size


# In[15]:


for i in range (255):
    print(nfl_data.columns[i])


# In[16]:


print("original dataset columns size %d\n" % nfl_data.shape[1])
print("after dropped cols with NaN columns size %d\n" % cols_with_NaN_dropped.shape[1])


# In[17]:


subset_nfl_data = nfl_data.loc[:, 'epa':'total_away_raw_yac_wpa'].head()
subset_nfl_data


# In[18]:


subset_nfl_data.fillna(0)


# In[19]:


subset_nfl_data.fillna(method='bfill',axis=0).fillna(0)


# In[20]:


from scipy import stats
from mlxtend.preprocessing import minmax_scaling
import seaborn as sns
import matplotlib.pyplot as plt
np.random.seed(0)


# In[21]:


original_data = np.random.exponential(size=1000)
scaled_data = minmax_scaling(original_data, columns=[0])

fig,ax = plt.subplots(1,2)
sns.distplot(original_data, ax=ax[0])
ax[0].set_title("Original data")

sns.distplot(scaled_data, ax=ax[1])
ax[1].set_title("Scaled data")


# In[22]:


normalized_data = stats.boxcox(original_data)
print(type(normalized_data))


fig,ax = plt.subplots(1,2)
sns.distplot(original_data, ax=ax[0])
ax[0].set_title("Original data")
sns.distplot(normalized_data[0], ax=ax[1])
ax[1].set_title("Normalized data")


# In[26]:


nfl_data.to_csv('NFL_Data_Cleaned.csv')

