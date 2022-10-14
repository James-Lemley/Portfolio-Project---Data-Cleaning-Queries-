#!/usr/bin/env python
# coding: utf-8

# In[6]:


# Import libraries

import pandas as pd
import seaborn as sns
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from matplotlib.pyplot import figure

get_ipython().run_line_magic('matplotlib', 'inline')
matplotlib.rcParams['figure.figsize']=(12,8) # Adjusts the configuration of the plots we will create

# Read in the data

df=pd.read_csv(r'C:\Users\jleml\Desktop\movies.csv')


# In[5]:


# Look at the data

df.head()


# In[91]:


# Look for any missing data

for col in df.columns:
    pct_missing = np.mean(df[col].isnull())
    print('{}-{}%'.format(col, pct_missing*100))


# In[92]:


# Data types for our columns

print(df.dtypes)


# In[43]:


# change data type of columns

df['gross'] = df['gross'].fillna(0).astype('int64')

df['budget'] = df['budget'].fillna(0).astype('int64')


# In[44]:


df


# In[40]:


# Create correct year column

df['yearcorrect'] = df['released'].astype(str).str.split(', ').str[-1].astype(str).str[:4]


# In[42]:


df


# In[47]:


df.sort_values(by=['gross'], inplace=False, ascending=False)


# In[46]:


pd.set_option('display.max_rows', None)


# In[50]:


# Drop duplicates

df.drop_duplicates()


# In[97]:


# Scatter plot with budget vs gross

plt.scatter(x=df['budget'], y=df['gross'])

plt.title('Budget vs Gross Earnings')

plt.xlabel('Gross Earnings')
plt.ylabel('Budget for Film')
plt.show()


# In[55]:


df.head()


# In[98]:


# Plot budget vs gross using seaborn

sns.regplot(x='budget', y='gross', data=df, scatter_kws={"color":"red"}, line_kws={"color":"blue"})


# In[99]:


# Let's start looking at correlation

df.corr(method='pearson') 


# In[ ]:


# High correlation between budget and gross


# In[100]:


correlation_matrix = df.corr(method='pearson')

sns.heatmap(correlation_matrix, annot=True)

plt.title('Correlation Matrix for Numeric Features')

plt.xlabel('Movie Features')

plt.ylabel('Movie Features')

plt.show()


# In[67]:


# Look at Company

df.head()


# In[101]:


df_numerized = df

for col_name in df_numerized.columns:
    if(df_numerized[col_name].dtype=='object'):
        df_numerized[col_name] = df_numerized[col_name].astype('category')
        df_numerized[col_name] = df_numerized[col_name].cat.codes
        
df_numerized


# In[88]:


df_numerized.corr(method='pearson')


# In[104]:


correlation_matrix = df_numerized.corr(method='pearson')

sns.heatmap(correlation_matrix, annot=True)

plt.title('Correlation Matrix for Numeric Features')

plt.xlabel('Movie Features')

plt.ylabel('Movie Features')

plt.show()


# In[82]:


correlation_mat = df_numerized.corr()

corr_pairs = correlation_mat.unstack()

corr_pairs


# In[83]:


sorted_pairs = corr_pairs.sort_values()

sorted_pairs


# In[85]:


high_corr = sorted_pairs[(sorted_pairs) > 0.5]

high_corr


# In[ ]:


# Votes and budget have the highest correlation to gross earnings
# Company has low correlation

