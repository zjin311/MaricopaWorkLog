

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
path='C:/Users/Mr.Goldss/Desktop/health care digital ticket EDA/'
filename='7-6-2020 Facility Itemized Tracker (1)'
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




#%%

#glove total by size and perc

total_glove=sum(df_clean_EMS[df_clean_EMS['Type']=='gloves'].groupby(['Type','size'])['Total Amount'].sum())

df_clean_EMS[df_clean_EMS['Type']=='gloves'].groupby(['Type','size'])['Total Amount'].apply(lambda x:x.sum()/total_glove)

# %%
#gowns
total_gowns=sum(df_clean_EMS[df_clean_EMS['Type']=='gowns'].groupby(['Type','size'])['Total Amount'].sum())

df_clean_EMS[df_clean_EMS['Type']=='gowns'].groupby(['Type','size'])['Total Amount'].apply(lambda x:x.sum()/total_gowns)


# %%

def what_perc_to_order(t):
    total=sum(df_clean_EMS[df_clean_EMS['Type']==str(t)].groupby(['Type','size'])['Total Amount'].sum())
    perc1=df_clean_EMS[df_clean_EMS['Type']==str(t)].groupby(['Type','size'])['Total Amount'].apply(lambda x:x.sum()/total)
    return 'total give out {type:} amount is {amount:}'.format(type=t,amount=total) ,  perc1


# %%
unique_type=list(df_clean_EMS['Type'].unique())
def find_type_size_prec():

    for i in unique_type:
        a=what_perc_to_order(i)
    return a

         

# %%
what_perc_to_order('gloves')

# %%
#df_clean_EMS.head()
'''2020-06-11 to 2020-06-25'''

'''

#df_clean_EMS[df_clean_EMS['Delivered Date'].between_time('2020-06-11','2020-06-18')]
c1=df_clean_EMS['Delivered Date']>='2020-06-11'
c2=df_clean_EMS['Delivered Date']<='2020-06-25'
andy1125=df_clean_EMS[c1 & c2].groupby('Type')['Total Amount'].sum()

andy1125.to_csv('from Jun 11 to 25.csv')
'''
#%%
'''reusable gowns''' 


df_reusable=df_clean_EMS[df_clean_EMS['Item Description'].str.contains('reusable')==True]



#%%
'''All Total'''
new_total=df_clean_EMS.groupby('Type')['Total Amount'].sum()

new_total.to_csv('{}total.csv'.format(date.today()))
# %%


          ''' Andy: average PPE supply each batch '''


def supply_per_batch():
    num_unique_batch=df_clean_EMS[df_clean_EMS['Facility Type']!='Acute'].groupby(['Delivered Date','Facility Name'])['Total Amount'].sum().shape[0]

    batch_avg=pd.DataFrame(df_clean_EMS[df_clean_EMS['Facility Type']!='Acute'].groupby(['Type'])['Total Amount'].sum()/num_unique_batch)

    return batch_avg['Total Amount'].apply(int).sort_values(ascending=False)

# %%supply_per_batch
print(supply_per_batch())
#%%


# %%


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



# %%
avg_ppe_by_facility_type(df_clean_EMS)


# %%
'''how many gloves we give out from Jun 4 to Jun 25 on average

'''

def avg_product_giveout(date1,date2):
    cond1=(df_clean_EMS['Delivered Date']>=date1)
    cond2=(df_clean_EMS['Delivered Date']<=date2)
    temp_df=df_clean_EMS[cond1 & cond2 ].groupby(['Type','size'])['Total Amount'].sum()/3
    temp_df=temp_df.apply(lambda x: int(x))
    return temp_df
# %%
temp_df=pd.DataFrame(avg_product_giveout('2020-06-04' , '2020-06-25'))
temp_df.sort_values(by=['Type','Total Amount'],ascending=False,inplace=True )
temp_df.to_csv('_Jun-4 to -25 product give out average by size.csv')

# %%
temp_df.reset_index(inplace=True)
#%%

# %%


# %%
df_clean_EMS.groupby(['Type'])['Total Amount'].sum().sort_values(ascending=False)

# %%
df_clean_EMS['Total Amount'].sum()


# %%

'''visualization'''

fig, (ax1, ax2) = plt.subplots(nrows = 1, ncols = 2, figsize = (13, 5))
sns.catplot(x="Type", y="Total Amount", hue="size", data=df_clean_EMS,
                height=6, kind="bar", palette="muted",ax=ax1)

ax1.set_xlabel('Type', fontsize=10)
ax1.set_ylabel('Total Amount', fontsize=10)
ax1.set_title('Calls Distribution', fontsize=10)
ax1.tick_params(labelsize=10)

sns.jointplot(x='Total Amount', y='week_of_year', kind="hex", color="#4CB391",data=df_clean_EMS,ax=ax2)

ax2.set_xlabel('Total', fontsize=10)
ax2.set_ylabel('week_of_year', fontsize=10)
ax2.set_title('total given by week', fontsize=10)
ax2.tick_params(labelsize=10)

plt.subplots_adjust(wspace=0.5)
plt.tight_layout() 
# %%
import calendar 

print(calendar.calendar(2020))
# %%
sns.barplot(x='Type', y='Total Amount', palette="rocket",data=df_clean_EMS)

# %%
