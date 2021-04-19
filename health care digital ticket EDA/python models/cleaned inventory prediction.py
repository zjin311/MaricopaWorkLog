

#%%  Read data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt 
from datetime import datetime
from datetime import date
path='C:/Users/Mr.Goldss/Desktop/health care digital ticket EDA/'
filename='6_23_2020 Facility Itemized Tracker (1)'
#%matplotlib inline
#%%

'''Read Data'''
def read_data(path,filename):
    df=pd.read_excel(path+filename+'.xlsx',sheet_name='New ALL')
    return df
read_data(path, filename)

#%% 
'''fix basic formatting----> df_clean'''
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
formatting_preprocessing(df)


#%% 
'''Drop EMS between 4/2/2020 and 4/30/2020 except the summary ticket'''
def drop_special_Ems():


    del_cond1=(df_clean['Delivered Date']>='2020-04-01') & (df_clean['Delivered Date']<='2020-04-30')
    del_cond2=(df_clean['Facility Type'] =='EMS') & (df_clean['Facility Name'] !='Phoenix Fire Resource Management')
    to_be_drop_EMS=df_clean.loc[del_cond1 & del_cond2]
    Ems_drop_list=to_be_drop_EMS.index
    df_clean_copy=df_clean
    df_clean_EMS=df_clean_copy.drop(Ems_drop_list)
    return df_clean_EMS

drop_special_Ems()

#%%
'''take care of sting formating and return df back'''

def str_format():
#fill the only missing qty per uom to be 1: base on reality

    df_clean_EMS['QTY Per UOM']=df_clean_EMS['QTY Per UOM'].fillna(1)

#replace 'Surgical Masks' to 'Surgical Mask'

    df_clean_EMS['Type']=df_clean_EMS['Type'].str.replace('Surgical Masks','Surgical Mask')
    df_clean_EMS['Type']=df_clean_EMS['Type'].str.lower()
    df_clean_EMS['Facility Type']=df_clean_EMS['Facility Type'].str.lower()
    df_clean_EMS['Facility Type']=df_clean_EMS['Facility Type'].str.strip()
    df_clean_EMS['Facility Type']=df_clean_EMS['Facility Type'].str.capitalize()


    to_replace={'Lctf':'Ltcf','Ltfc':'Ltcf'}
    for key in to_replace.keys():
        df_clean_EMS['Facility Type']=df_clean_EMS['Facility Type'].str.replace(key,to_replace[key])


    df_clean_EMS['Facility Type']=df_clean_EMS['Facility Type'].str.replace('Acute care','Acute')

    df_clean_EMS['Facility Name']=df_clean_EMS['Facility Name'].str.lower()

    return df_clean_EMS
str_format()

#%%
'''
12 weeks average PPE Burn rate

create a new feature: week of the year based on Delivered data column
'''
def take_care_date():


    df_clean_EMS['week_of_year']=df_clean_EMS['Delivered Date'].dt.strftime('%U')

    return df_clean_EMS

take_care_date()

#%%


#-------------------------------------------------------------
#-------------------------------------------------------------
#-------------------------------------------------------------
#-------------------------------------------------------------
#-------------------------------------------------------------









'''Analysis Part'''
'''all time high per type by week: used as PPE buying strategy'''

#df_clean_EMS[df_clean_EMS['Type']=='gloves'].groupby(['week_of_year'])['Type','Total Amount'].agg(['first','sum']).sort_values(by='Total Amount',ascending=False)

#%%

# maximum number by different week

type_unique=df_clean_EMS.Type.unique()
maxnumber=[]
for i in type_unique:
    maxnumber.append(df_clean_EMS[df_clean_EMS['Type']==i].groupby(['week_of_year','Type'])['Total Amount'].agg('sum').sort_values(ascending=False)[0])

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



#-------------------------------------------------------------
#-------------------------------------------------------------
#-------------------------------------------------------------
#-------------------------------------------------------------
#-------------------------------------------------------------




#%%

#6/18/2020
'''how many ppe we give per ticket by different facility type'''
def avg_ppe_by_facility_type(df_clean_EMS):


    unique_faclity_type=df_clean_EMS['Facility Type'].unique()
    facility_type_count=[]

    for i in range(len(unique_faclity_type)):
        #with shape func to calculate count of unique facility in different facility type
        facility_type_count.append(df_clean_EMS[df_clean_EMS['Facility Type']==unique_faclity_type[i]].groupby(['Delivered Date','Facility Name'])['Total Amount'].sum().shape[0])
    
    
    unique_facility_type_count=dict(zip(unique_faclity_type,facility_type_count))

    for key,value in unique_facility_type_count.items():
        print(key)

        print(df_clean_EMS[df_clean_EMS['Facility Type']==key].groupby(['Type'])['Total Amount'].sum()/value)
        print('-----------------')
        print()



avg_ppe_by_facility_type(df_clean_EMS)


#%%
'''6/25/2020 Andy: how many large glove we give out per week'''
df_clean_EMS['Item Description']=df_clean_EMS['Item Description'].str.lower()

s=[]
m=[]
l=[]
u=[]
other=[]

for i in df_clean_EMS['Item Description']:

        if 'small' in i:
            s.append(i)
        elif 'medium' in i:
            m.append(i)
        elif 'large' in i:
            l.append(i)
        elif 'universal' in i:
            u.append(i)
        else:
            other.append(i)
    


        
#%%
  #   
    df_clean_EMS['size']=df_clean_EMS['Item Description'].replace(dict.fromkeys(s,'small'))

    df_clean_EMS['size']=df_clean_EMS['Item Description'].replace(dict.fromkeys(m,'medium'))

    df_clean_EMS['size']=df_clean_EMS['Item Description'].replace(dict.fromkeys(l,'large'))

    df_clean_EMS['size']=df_clean_EMS['Item Description'].replace(dict.fromkeys(u,'universal'))

    df_clean_EMS['size']=df_clean_EMS['Item Description'].replace(dict.fromkeys(other,'other'))


#%%

''''''





df_clean_EMS['size']=df_clean_EMS['Item Description'].apply(
    lambda x:x.replace(x,'small') if x in s else(x.replace(x,'medium') if x in m else( x.replace(x,'large') if x in l else x.replace(x,'universal')) ))


# %%

#glove total by size and perc

total_glove=sum(df_clean_EMS[df_clean_EMS['Type']=='gloves'].groupby(['Type','size'])['Total Amount'].sum())

df_clean_EMS[df_clean_EMS['Type']=='gloves'].groupby(['Type','size'])['Total Amount'].apply(lambda x:x.sum()/total_glove)

# %%
#gowns
total_gowns=sum(df_clean_EMS[df_clean_EMS['Type']=='gowns'].groupby(['Type','size'])['Total Amount'].sum())

df_clean_EMS[df_clean_EMS['Type']=='gowns'].groupby(['Type','size'])['Total Amount'].apply(lambda x:x.sum()/total_gowns)














# %%
