#%%  Read data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt 
from datetime import datetime
from datetime import date
#%matplotlib inline
#%%
path='C:/Users/Mr.Goldss/Desktop/health care digital ticket EDA/richard inventory report/'
filename='Inventory.2020-07-20.104622001'
df=pd.read_excel(path+filename+'.xlsx')

df.shape

# %%
df_txt=df[['Article','Category', 'Description','Qty Avail.']]

# %%
