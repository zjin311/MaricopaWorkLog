# -*- coding: utf-8 -*-
"""
Created on Mon May 11 14:10:57 2020

@author: Mr.Goldss
"""

import pandas as pd
import numpy as np

#%%  Read data
df2=pd.read_excel('C:/Users/Mr.Goldss/Desktop/health care digital ticket EDA/Facility Itemized Tracker _all.xlsx'
                 ,sheet_name='Itemized Tracker')

#preprocessing
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
    df=df[df['Units']>0]
# fill missing product source with SNS
    df['Product Source']=df['Product Source'].fillna('SNS')
# fill missing cheat sheet with solid
    df['Cheat Sheet']=df['Cheat Sheet'].fillna('Good')

    df_clean=df
    return df_clean


df_clean=formatting_preprocessing(df2)

#%% additional cleaning
df_clean=df_clean['Type'].str.replace('Surgical Mask','Surgical Masks')


#%% EDA
#1. explore type and total amount relationship
df2.groupby('Type')['Total Amount'].sum().sort_values(ascending=False)


#2. explore facility/type/total amount
df_clean.groupby(['Facility Name','Type')['Total Amount'].sum().sort_values(ascending=False)

#3. explore source/ amount
df_clean.groupby(['Product Source','Type'])['Total Amount'].sum()

#4. explore facility type / type/ amount
df_clean.groupby(['Facility Type','Type'])['Total Amount'].sum()

