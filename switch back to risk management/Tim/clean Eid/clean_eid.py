#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime 

#%%
df=pd.read_excel('SSNEmpolyees20210323c7675fd5fec04571ae963369e9e53bc2.xlsx')
# %%

df.info()
#%%
'''Data prepare'''
df=df[df['Social Security']!='xxx-xx-0000']
df=df[df['Social Security'].notna()]
df['end_with_asterisk']=df['EmployeeNumber'].str.endswith('*')
df['EmployeeNumber']=df['EmployeeNumber'].str.replace('*','')
df['eid_last4']=df['EmployeeNumber'].apply(lambda x:str(x)[-4:])
df['ssn_last4']=df['Social Security'].apply(lambda x:str(x)[-4:])
df['Eid_SSN']=df['eid_last4']==df['ssn_last4']
df['ssn_fn_ln']=df['Social Security']+df['First Name']+df['Last Name']

#%%
# search by name
df[(df['First Name']=='David')&(df['Last Name']=='Yevin')]
#%%

df_groupby=pd.DataFrame(df.groupby(['Social Security','First Name','Last Name'])['EmployeeNumber'].count().reset_index())
df_groupby['ssn_fn_ln']=df_groupby['Social Security']+df_groupby['First Name']+df_groupby['Last Name']
df_groupby2=df_groupby[df_groupby['EmployeeNumber']>=2]
df_groupby1=df_groupby[df_groupby['EmployeeNumber']==1]
#%%
df_one_match=df[df['ssn_fn_ln'].isin(df_groupby1['ssn_fn_ln']) & (df['Eid_SSN']==True)]

df_more_match=df[df['ssn_fn_ln'].isin(df_groupby2['ssn_fn_ln']) & (df['Eid_SSN']==True)]





#%%
#find records with duplicate records
dup_ssn=pd.DataFrame(df['Social Security'].value_counts()>1)
dup_ssn=dup_ssn[dup_ssn['Social Security']==True].reset_index()
dup_ssn.columns=['ssn','dup_ssn']
#%%
test=df[(df['Social Security'].isin(dup_ssn['ssn']))]

#%%
df[(df['Social Security'].isin(dup_ssn['ssn'])) & (df['Eid_SSN']==False)]


# %%
df_t=df[df['last 4']==True]
df_f=df[df['last 4']==False]
# %%
'''step 1'''
#t_ssn store all ssn with dupilicate
#analysis records with ssn and eid
t_ssn=pd.DataFrame(df['Social Security'].value_counts()>1)
t_ssn=t_ssn.reset_index()
t_ssn.columns=['SSN','Social Security']
t_ssn=t_ssn[t_ssn['Social Security']==True]
print(t_ssn)


#%%
#one ssn come with multiple records
df_duplicate=df[df['Social Security'].isin(t_ssn['SSN'])]
# %%
df_duplicate.sort_values('Social Security')
# %%
#we can do this cuz there is no missing value in first and last name column
a=df_duplicate.groupby(['Social Security','First Name','Last Name'])['EmployeeNumber'].count()
a=pd.DataFrame(a).reset_index()
# %%
a=a[a['EmployeeNumber']>1]
a.to_excel('same ssn with more than 1 Eid.xlsx')
# %%
'''step 2'''
#just 1 ssn record but eid may == ssn
df_one=df[~df['Social Security'].isin(t_ssn['SSN'])]
# %%
# we don't care records without ssn
df_one=df_one[df_one['Social Security'].notna()]
# %%
b=df_one[df_one['Eid_SSN']==True]
b.to_excel('last 4 digit match ssn_Eid only 1 matching.xlsx')
# %%
