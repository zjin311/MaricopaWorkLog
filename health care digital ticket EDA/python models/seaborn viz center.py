import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt 
from datetime import datetime

#%config InlineBackend.figure_format = 'retina'
#%matplotlib inline
#%%
path='C:/Users/Mr.Goldss/Desktop/health care digital ticket EDA/'
filename='7_23_2020 Facility Itemized Tracker (1)'
df=pd.read_excel(path+filename+'.xlsx',sheet_name='New ALL')

df.shape

#%%                     fix basic formatting: preprocessing
'''data preprocessing'''

def formatting_preprocessing(df):
    want_col=['Date', 'Facility Name',
              'Facility Type', 'Item', 'Product Source', 'Type', 'UOM',
              'Qty', 'Total']
    df=df[want_col]
    df.columns=['Delivered Date', 'Facility Name',
       'Facility Type', 'Item Description', 'Product Source', 'Type', 'Units',
       'QTY Per UOM', 'Total Amount']
    df=df[df['Delivered Date'].notna()]
    df=df[df['Item Description'].notna()]
    df=df[(df['Units']>0) & (df['Units'].notna())]
    df['Product Source']=df['Product Source'].fillna('SNS')
    df_clean=df
    return df_clean
df_clean=formatting_preprocessing(df)

def del_wrong_ems(df_clean):

    del_cond1=(df_clean['Delivered Date']>='2020-04-01') & (df_clean['Delivered Date']<='2020-04-30')
    del_cond2=(df_clean['Facility Type'] =='EMS') & (df_clean['Facility Name'] !='Phoenix Fire Resource Management')
    to_be_drop_EMS=df_clean.loc[del_cond1 & del_cond2]
    Ems_drop_list=to_be_drop_EMS.index
    df_clean_copy=df_clean
    df_clean_EMS=df_clean_copy.drop(Ems_drop_list)
    return df_clean_EMS
df_clean_EMS=del_wrong_ems(df_clean)

def deep_clean_str_manipulate(df_clean_EMS):

    df_clean_EMS['QTY Per UOM']=df_clean_EMS['QTY Per UOM'].fillna(1)
    df_clean_EMS['Type']=df_clean_EMS['Type'].str.replace('Surgical Masks','Surgical Mask')
    df_clean_EMS['Type']=df_clean_EMS['Type'].str.lower()
    df_clean_EMS.groupby('Type')['Total Amount'].sum()

    df_clean_EMS['Facility Type']=df_clean_EMS['Facility Type'].str.lower()
    df_clean_EMS['Facility Type']=df_clean_EMS['Facility Type'].str.strip()
    df_clean_EMS['Facility Type']=df_clean_EMS['Facility Type'].str.capitalize()


    to_replace={'Lctf':'Ltcf','Ltfc':'Ltcf'}
    for key in to_replace.keys():
        df_clean_EMS['Facility Type']=df_clean_EMS['Facility Type'].str.replace(key,to_replace[key])


    df_clean_EMS['Facility Type']=df_clean_EMS['Facility Type'].str.replace('Acute care','Acute')

    df_clean_EMS['Item Description']=df_clean_EMS['Item Description'].str.lower()

    s=[]
    m=[]
    l=[]
    other=[]

    for i in df_clean_EMS['Item Description']:

        if 'small' in i:
            s.append(i)
        elif 'medium' in i:
            m.append(i)
        elif 'large' in i:
            l.append(i)
        else:
            other.append(i)
    df_clean_EMS['size']=df_clean_EMS['Item Description'].apply(
        lambda x:x.replace(x,'small') if x in s 
        else(x.replace(x,'medium') if x in m else( x.replace(x,'large') if x in l else x.replace(x,'universal')) ))


    df_clean_EMS['week_of_year']=df_clean_EMS['Delivered Date'].dt.strftime('%U')
    df_clean_EMS['Month']=df_clean_EMS['Delivered Date'].dt.month
    df_clean_EMS['Total Amount']=df_clean_EMS['Total Amount'].astype(int)
    return df_clean_EMS
deep_clean_str_manipulate(df_clean_EMS)
#%%
'''countplot'''
fig, ax1, = plt.subplots()
fig.set_size_inches(25, 12)
sns.countplot(x='Facility Type',data=df_clean_EMS,ax=ax1)
plt.xticks(fontsize=18,rotation=90)
plt.xlabel('Facility', fontsize=15)
plt.ylabel('count', fontsize=20)
plt.title('Beltmann Inventory Distribution', fontsize=30)

plt.tight_layout() 
plt.show()
#%%
#,barplot with plt.text to label all columns'''
'''
for index, value in enumerate(df_clean_EMS['Total Amount']):
    plt.text(index,value+400, str(value),fontsize=12)
plt.tight_layout() 
'''

# %%
'''sns.lineplot---->date as x axis'''
#we only care about 5 product type
type_needed=['n95','surgical mask','gowns','gloves']
fig, ax1, = plt.subplots()
fig.set_size_inches(25, 12)
sns.lineplot(x='Month',y='Total Amount',hue='Type',\
    data=df_clean_EMS[df_clean_EMS['Type'].isin(type_needed)],ax=ax1,\
        ci=False, markers=True, style='Type')
plt.xticks(fontsize=18,rotation=90)
plt.xlabel('Month', fontsize=20)
plt.ylabel('Amount', fontsize=25)
plt.title('day', fontsize=30)

plt.tight_layout() 
plt.show()
#%%
type_needed=['n95','surgical mask','gowns','gloves']
fig, ax1, = plt.subplots()
fig.set_size_inches(25, 12)
sns.relplot(x='Month',y='Total Amount',hue='Type',\
    data=df_clean_EMS[df_clean_EMS['Type'].isin(type_needed)],ax=ax1,\
        ci=False, markers=True, style='Type',kind='line')
plt.xticks(fontsize=18,rotation=90)
plt.xlabel('Month', fontsize=20)
plt.ylabel('Amount', fontsize=25)
plt.title('day', fontsize=30)

plt.tight_layout() 
plt.show()


# %%
'''sns.barplot()'''
#one thing need to realize is that the bar is not the sum but the mean

type_needed=['n95','surgical mask','gowns','gloves']
fig, ax1, = plt.subplots()
fig.set_size_inches(25, 12)
sns.barplot(x='Month',y='Total Amount',hue='Type',\
    data=df_clean_EMS[df_clean_EMS['Type'].isin(type_needed)],ax=ax1,\
        ci=False)
plt.yticks(fontsize=18)
plt.xticks(fontsize=20)
plt.xlabel('Month of the year', fontsize=20)
plt.ylabel('mean Amount', fontsize=25)
plt.title('day', fontsize=30)

plt.tight_layout() 
plt.show()
# %%
'''switch x and y and use orient=h to transpose '''
type_needed=['n95','surgical mask','gowns','gloves']
fig, ax1, = plt.subplots()
fig.set_size_inches(25, 12)
sns.barplot(y='Month',x='Total Amount',hue='Type',\
    data=df_clean_EMS[df_clean_EMS['Type'].isin(type_needed)],ax=ax1,\
        ci=False,orient='h')
plt.yticks(fontsize=18)
plt.xticks(fontsize=20)
plt.xlabel('Month of the year', fontsize=20)
plt.ylabel('mean Amount', fontsize=25)
plt.title('day', fontsize=30)

plt.tight_layout() 
plt.show()


# %%
'''how can i change the color?'''
#go to seaborn official web to find color option
# search for html color code--> find the (hex) code like this :#E315C7 
type_needed=['n95','surgical mask','gowns','gloves']
fig, ax1, = plt.subplots()
fig.set_size_inches(25, 12)
sns.barplot(y='Month',x='Total Amount',hue='Type',\
    data=df_clean_EMS[df_clean_EMS['Type'].isin(type_needed)],ax=ax1,\
        ci=False,orient='h',color='#E315C7')
plt.yticks(fontsize=18)
plt.xticks(fontsize=20)
plt.xlabel('Month of the year', fontsize=20)
plt.ylabel('mean Amount', fontsize=25)
plt.title('day', fontsize=30)

plt.tight_layout() 
plt.show()


# %%
'''hist is a good way to analysis the distribution of the data'''

fig, ax1, = plt.subplots()
fig.set_size_inches(25, 12)
sns.distplot(df_clean_EMS[(df_clean_EMS['Type']=='gloves')\
    &(df_clean_EMS['Facility Type']=='Ltcf')\
        &(df_clean_EMS['Total Amount']<=500)]['Total Amount'],ax=ax1)
plt.yticks(fontsize=18)
plt.xticks(fontsize=20)
plt.xlabel('give amount', fontsize=20)
plt.ylabel('count', fontsize=25)
plt.title('give out distribution', fontsize=30)

mean=df_clean_EMS[(df_clean_EMS['Type']=='gloves')\
    &(df_clean_EMS['Facility Type']=='Ltcf')\
        &(df_clean_EMS['Total Amount']<=500)]['Total Amount'].mean()
plt.axvline(mean, color='red')

plt.tight_layout() 
plt.show()


# %%
def draw_type_facility_type_amount_dist(Type,FacilityType,upper_level=50000):

    fig, ax1, = plt.subplots()
    fig.set_size_inches(25, 12)
    sns.distplot(df_clean_EMS[(df_clean_EMS['Type']==Type)\
        &(df_clean_EMS['Facility Type']==FacilityType)\
            &(df_clean_EMS['Total Amount']<=upper_level)]['Total Amount'],ax=ax1)
    plt.yticks(fontsize=18)
    plt.xticks(fontsize=20)
    plt.xlabel('give amount', fontsize=20)
    plt.ylabel('count', fontsize=25)
    plt.title('give out distribution', fontsize=30)

    mean=df_clean_EMS[(df_clean_EMS['Type']==Type)\
        &(df_clean_EMS['Facility Type']==FacilityType)\
            &(df_clean_EMS['Total Amount']<=upper_level)]['Total Amount'].mean()
    plt.axvline(mean, 0,1,color='red')

    plt.tight_layout() 
    plt.show()

# %%
draw_type_facility_type_amount_dist('n95','Acute',2000)

# %%
'''box plot is another good way to show distribution'''
fig, ax1, = plt.subplots()
fig.set_size_inches(25, 12)
sns.boxplot(df_clean_EMS[(df_clean_EMS['Type']=='gloves')\
    &(df_clean_EMS['Facility Type']=='Ltcf')\
        &(df_clean_EMS['Total Amount']<=5000)]['Total Amount'],ax=ax1)
plt.yticks(fontsize=18)
plt.xticks(fontsize=20)
plt.xlabel('give amount', fontsize=20)
plt.ylabel('count', fontsize=25)
plt.title('give out distribution', fontsize=30)

mean=df_clean_EMS[(df_clean_EMS['Type']=='gloves')\
    &(df_clean_EMS['Facility Type']=='Ltcf')\
        &(df_clean_EMS['Total Amount']<=500)]['Total Amount'].mean()
plt.axvline(mean, color='red')

plt.tight_layout() 
plt.show()


# %%
fig, ax1, = plt.subplots()
#fig.set_size_inches(25, 12)
sns.set(rc={'figure.figsize':(12,10)})
sns.boxplot(y='Total Amount',
            x='Month',data=df_clean_EMS,ax=ax1)
plt.tight_layout() 
plt.show()



# %%
'''sns.swarmplot(), which can show every data point with a distribution,
 is a good helper for sns.boxplot, '''


fig, ax1, = plt.subplots()
#fig.set_size_inches(25, 12)
sns.set(rc={'figure.figsize':(12,10)})
sns.boxplot(x='Total Amount',
            y='Month',data=df_clean_EMS[df_clean_EMS['Total Amount']<=5000],ax=ax1,orient='h',color='#E55C94')

sns.swarmplot(x='Total Amount',
            y='Month',data=df_clean_EMS[df_clean_EMS['Total Amount']<=5000],ax=ax1,orient='h',color='green')
plt.tight_layout() 
plt.show()


# %%
'''same as sns.violinplot'''
fig, ax1, = plt.subplots()
#fig.set_size_inches(25, 12)
#sns.set(rc={'figure.figsize':(12,10)})
sns.boxplot(x='Total Amount',
            y='Month',data=df_clean_EMS[df_clean_EMS['Total Amount']<=5000],ax=ax1,orient='h',color='#E55C94')

sns.violinplot(x='Total Amount',
            y='Month',data=df_clean_EMS[df_clean_EMS['Total Amount']<=5000],ax=ax1,orient='h',color='green')
plt.tight_layout() 
plt.show()



# %%
'''let's try sns.scatterplot()'''
fig, ax1, = plt.subplots()
#fig.set_size_inches(25, 12)
#sns.set(rc={'figure.figsize':(12,10)})
sns.scatterplot(y='Total Amount',
            x='week_of_year',data=df_clean_EMS[df_clean_EMS['Total Amount']<=5000],\
                ax=ax1,color='#E55C94')
plt.tight_layout() 
plt.show()


# %%
'''what is lmplot()==> basically: scattern plot with regression line'''
#this is not a good example, basically you need to find 2 continue variable


sns.lmplot(y='Total Amount',
            x='Month',data=df_clean_EMS[df_clean_EMS['Total Amount']<=5000],height=10)
            
plt.tight_layout() 
plt.show()