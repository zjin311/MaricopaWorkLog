

#%%  Read data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt 
from datetime import datetime
from datetime import date
#%matplotlib inline
#%%
path='C:/Users/Mr.Goldss/Desktop/health care digital ticket EDA/'
filename='6_23_2020 Facility Itemized Tracker (1)'
df=pd.read_excel(path+filename+'.xlsx',sheet_name='New ALL')

df.shape

#%% fix basic formatting
def formatting_preprocessing(df):
    want_col=['Date', 'Facility Name',
              'Facility Type', 'Item', 'Product Source', 'Type', 'UOM',
              'Qty', 'Total']
    df=df[want_col]
#     rename and change format
    df.columns=['Delivered Date', 'Facility Name',
       'Facility Type', 'Item Description', 'Product Source', 'Type', 'Units',
       'QTY Per UOM', 'Total Amount']

#drop first 5 rows that are empty
    #df=df.iloc[5:]

#drop rows with missing date----> empty
    df=df[df['Delivered Date'].notna()]
#drop rows with missing item desc
    df=df[df['Item Description'].notna()]
# drop rows with units assigned =0
    df=df[(df['Units']>0) & (df['Units'].notna())]
# fill missing product source with SNS
    df['Product Source']=df['Product Source'].fillna('SNS')
# fill missing cheat sheet with solid
    #f['Cheat Sheet']=df['Cheat Sheet'].fillna('Good')

    df_clean=df
    return df_clean
#%% Apply the general preprocessing function
df_clean=formatting_preprocessing(df)









#%% Drop EMS between 4/2/2020 and 4/30/2020 except the summary ticket


del_cond1=(df_clean['Delivered Date']>='2020-04-01') & (df_clean['Delivered Date']<='2020-04-30')
del_cond2=(df_clean['Facility Type'] =='EMS') & (df_clean['Facility Name'] !='Phoenix Fire Resource Management')
to_be_drop_EMS=df_clean.loc[del_cond1 & del_cond2]
Ems_drop_list=to_be_drop_EMS.index
df_clean_copy=df_clean
df_clean_EMS=df_clean_copy.drop(Ems_drop_list)



#%%


#fill the only missing qty per uom to be 1: base on reality

df_clean_EMS['QTY Per UOM']=df_clean_EMS['QTY Per UOM'].fillna(1)

#replace 'Surgical Masks' to 'Surgical Mask'

df_clean_EMS['Type']=df_clean_EMS['Type'].str.replace('Surgical Masks','Surgical Mask')
df_clean_EMS['Type']=df_clean_EMS['Type'].str.lower()
df_clean_EMS.groupby('Type')['Total Amount'].sum()

#%%


df_clean_EMS['Facility Type']=df_clean_EMS['Facility Type'].str.lower()
df_clean_EMS['Facility Type']=df_clean_EMS['Facility Type'].str.strip()
df_clean_EMS['Facility Type']=df_clean_EMS['Facility Type'].str.capitalize()


to_replace={'Lctf':'Ltcf','Ltfc':'Ltcf'}
for key in to_replace.keys():
    df_clean_EMS['Facility Type']=df_clean_EMS['Facility Type'].str.replace(key,to_replace[key])


df_clean_EMS['Facility Type']=df_clean_EMS['Facility Type'].str.replace('Acute care','Acute')

#%%
df_clean_EMS['Facility Name']=df_clean_EMS['Facility Name'].str.lower()
#%%
'''
'''
Horizon Health and Wellness
Horizon Health & Wellness
Horizon Health 

lower facility name
'''
df_clean_EMS['Facility Name']=df_clean_EMS['Facility Name'].str.lower()
horizon={'horizon health and wellness':'horizon health',
'horizon health & wellness':'horizon health'}

for k,v in horizon.items():

    df_clean_EMS['Facility Name']=df_clean_EMS['Facility Name'].str.replace(k,v)



want_list=['circle the city','horizon health','native health','valle del sol']


#%%

for i in want_list:
    print('Department: {}'.format(i.capitalize()))
    
    print(df_clean_EMS[df_clean_EMS['Facility Name']==i].groupby('Type')['Total Amount'].sum())


    print('------------------------------------------------')
    print('\n')






#%%
df_clean_EMS[df_clean_EMS['Facility Type']=='Ltcf'].groupby('Type')['Total Amount'].sum()
#%%
'''ALL'''
update_total_by_type=df_clean_EMS.groupby('Type')['Total Amount'].sum()




update_total_by_type.to_csv('update_total_by_type')


'''




















#%%

'''Analysis Gerry asked'''

df_clean_EMS[(df_clean_EMS['Facility Type']!='EMS') & (df_clean_EMS['Facility Type']!='Acute')].groupby('Type')['Total Amount'].sum()


xxx=df_clean_EMS[(df_clean_EMS['Facility Type']!='EMS') & (df_clean_EMS['Facility Type']!='Acute')]

#how many transaction we send without ems and acute
print(df_clean_EMS[df_clean_EMS['Facility Name'].isin(["Glencroft Center for Modern Aging","Sarah's Place at Glencroft",'Glencroft - Providence Place'])].groupby('Type')['Total Amount'].sum())

#%%

'''Matt request'''
print(df_clean_EMS[df_clean_EMS['Facility Name']=='Glencroft Center for Modern Aging'].groupby('Type')['Total Amount'].sum())
print(df_clean_EMS[df_clean_EMS['Facility Name']=="Sarah's Place at Glencroft"].groupby('Type')['Total Amount'].sum())
print(df_clean_EMS[df_clean_EMS['Facility Name']=='Glencroft - Providence Place'].groupby('Type')['Total Amount'].sum())

#%%










#%%
'''
12 weeks average PPE Burn rate

create a new feature: week of the year based on Delivered data column
'''
df_clean_EMS['week_of_year']=df_clean_EMS['Delivered Date'].dt.strftime('%U')

df_clean_EMS.head()


#%%


df_clean_EMS[(df_clean_EMS['Facility Type']!='EMS') & (df_clean_EMS['Facility Type']!='Acute')].groupby(['Facility Name','Delivered Date']).count()
#%%

#-------------------------------------------------------------









'''Analysis Part'''
'''all time high per type by week'''

#df_clean_EMS[df_clean_EMS['Type']=='gloves'].groupby(['week_of_year'])['Type','Total Amount'].agg(['first','sum']).sort_values(by='Total Amount',ascending=False)

#%%

# maximum number by different week
type_unique=df_clean_EMS.Type.unique()
maxnumber=[]
for i in type_unique:
    maxnumber.append(df_clean_EMS[df_clean_EMS['Type']==i].groupby(['week_of_year','Type'])['Total Amount'].agg('sum').sort_values(ascending=False)[0])

print(maxnumber)
#%%
# 2nd number by different week

type_unique=df_clean_EMS.Type.unique()
maxnumber=[]
for i in type_unique:
    maxnumber.append(df_clean_EMS[df_clean_EMS['Type']==i].groupby(['week_of_year','Type'])['Total Amount'].agg('sum').sort_values(ascending=False)[1])

print(maxnumber)



#%%
#create dictionary
alltimehighbytype=list(zip(type_unique,maxnumber))

print(alltimehighbytype)

df_ana=pd.DataFrame(alltimehighbytype,columns=['Type','All_Time_High_Weekly'])

#-------------------------------------------------------------
#%%
''' 95 & 99 confidence level calculation'''
mean_by_type=[]
for i in type_unique:
    mean_by_type.append(int(df_clean_EMS[df_clean_EMS['Type']==i]['Total Amount'].mean()))

all_time_mean_by_type=list(zip(mean_by_type,type_unique))


#add to df

df_ana['weekly_mean']=mean_by_type
print(all_time_mean_by_type)

#%%
'''median'''

median_by_type=[]
for i in type_unique:
    median_by_type.append(int(df_clean_EMS[df_clean_EMS['Type']==i]['Total Amount'].median()))



#%%
'''std'''
std_by_type=[]
for i in type_unique:
    std_by_type.append(df_clean_EMS[df_clean_EMS['Type']==i]['Total Amount'].agg(np.std))

df_ana['std']=std_by_type
df_ana['std']=df_ana['std'].fillna(0)
df_ana['std']=df_ana['std'].round()

#%%
# 95
df_ana['95 confident order amount']=df_ana['std']*2+df_ana['weekly_mean']
df_ana['99 confident order amount']=df_ana['std']*3+df_ana['weekly_mean']

df_ana['100(4std)confident order amount']=df_ana['std']*4+df_ana['weekly_mean']
#%%
#10 week total amount to order
df_ana['10 week order (all time high)']=df_ana['All_Time_High_Weekly']*10
df_ana['10 week order (99 confident)']=df_ana['99 confident order amount']*10
df_ana['10 week order (95 confident)']=df_ana['95 confident order amount']*10
df_ana['10 week order (4std confident)']=df_ana['100(4std)confident order amount']*10

#%%
df_ana['12 week order (all time high)']=df_ana['All_Time_High_Weekly']*12
df_ana['12 week order (99 confident)']=df_ana['99 confident order amount']*12
df_ana['12 week order (95 confident)']=df_ana['95 confident order amount']*12
df_ana['12 week order (4std confident)']=df_ana['100(4std)confident order amount']*12

#%%
df_ana.to_csv('PPE Order amount prediction with confidence level by zixiang michael Jin.csv')


#%%
#df_clean_EMS['month']=df_clean_EMS['Delivered Date'].dt.strftime('%m')
'''
DS_month_type_amount=df_clean_EMS.groupby(['month','Type'])['Total Amount'].sum()
DS_month_type_amount=pd.DataFrame(DS_month_type_amount)

'''
#%%
DS_month_type_amount.to_excel('DS_month_type_amount.xlsx')
#%% detailed cleaning
'''
#1. fix facility type
df_clean['Facility Type'].unique()

df_clean['Facility Type']=df_clean['Facility Type'].str.lower()
df_clean['Facility Type']=df_clean['Facility Type'].str.strip()
#df_clean['Facility Type']=df_clean['Facility Type'].str.capitalize()


to_replace={'Lctf':'Ltcf','Ltfc':'Ltcf'}
for key in to_replace.keys():
    df_clean['Facility Type']=df_clean['Facility Type'].str.replace(key,to_replace[key])


df_clean['Facility Type']=df_clean['Facility Type'].str.replace('Acute care','Acute')



#{'Surprise Health and Rehab Center':'Ltcf','Sunview Respiratory and Rehab':'?',}

#df["Facility Type"].fillna(df.groupby("Facility Name")["Facility Type"].transform(lambda x: x.mode()))



#%% EDA
#1. explore type and total amount relationship
df_clean.groupby('Type')['Total Amount'].sum().sort_values(ascending=False)


#2. explore facility/type/total amount
df_clean.groupby(['Facility Name','Type')['Total Amount'].sum().sort_values(ascending=False)

#3. explore source/ amount
df_clean.groupby(['Product Source','Type'])['Total Amount'].sum()

#4. explore facility type / type/ amount
df_clean.groupby(['Facility Type','Type'])['Total Amount'].sum()



'''






#%%

df_clean_EMS.to_csv('df_clean_EMS.csv')





















# %%
