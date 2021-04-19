
import pandas as pd
import numpy as np

#%%  Read data
df=pd.read_excel('C:/Users/Mr.Goldss/Desktop/health care digital ticket EDA/Jin cleaned Facility Itemized Tracker _all.xlsx'
                 ,sheet_name='Itemized Tracker')

#%% fix basic formatting
def formatting_preprocessing(df):
    want_col=['Delivered Date', 'Facility Name',
              'Facility Type', 'Item Description', ' Product Source', 'Type', 'Units',
              'QTY Per UOM', 'Total Amount', 'Cheat Sheet']
    df=df[want_col]
#     rename and change format
    df.columns=['Delivered Date', 'Facility Name',
       'Facility Type', 'Item Description', 'Product Source', 'Type', 'Units',
       'QTY Per UOM', 'Total Amount', 'Cheat Sheet']

#drop first 5 rows that are empty
    df=df.iloc[5:]

#drop rows with missing date----> empty
    df=df[df['Delivered Date'].notna()]
#drop rows with missing item desc
    df=df[df['Item Description'].notna()]
# drop rows with units assigned =0
    df=df[(df['Units']>0) & (df['Units'].notna())]
# fill missing product source with SNS
    df['Product Source']=df['Product Source'].fillna('SNS')
# fill missing cheat sheet with solid
    df['Cheat Sheet']=df['Cheat Sheet'].fillna('Good')

    df_clean=df
    return df_clean
#%% Apply the general preprocessing function
df_clean=formatting_preprocessing(df)



#%% detailed cleaning

#1. fix facility type
df_clean['Facility Type'].unique()

df_clean['Facility Type']=df_clean['Facility Type'].str.lower()
df_clean['Facility Type']=df_clean['Facility Type'].str.strip()
df_clean['Facility Type']=df_clean['Facility Type'].str.capitalize()


to_replace={'Lctf':'Ltcf','Ltfc':'Ltcf'}
for key in to_replace.keys():
    df_clean['Facility Type']=df_clean['Facility Type'].str.replace(key,to_replace[key])


df_clean['Facility Type']=df_clean['Facility Type'].str.replace('Acute care','Acute')



{'Surprise Health and Rehab Center':'Ltcf','Sunview Respiratory and Rehab':'?',}

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











