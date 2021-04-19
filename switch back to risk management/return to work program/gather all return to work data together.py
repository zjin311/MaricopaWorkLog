#%%
import pandas as pd
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
import seaborn as sns
import datetime 
import missingno as msno
import smartsheet
import os
import docx
datt=str(datetime.datetime.now())[:10]

PROJECT_ROOT_DIR = "."
CHAPTER_ID = "gather all data together"
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID)
os.makedirs(IMAGES_PATH, exist_ok=True)

def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)
# %%

'''Step 1: get transactional data for TTD and TPD'''
#what's the transactional dataset filter??
#date>=2016/1/1  for WC only for TTD and TPD financial cat only
def read_transaction_database(filename):
    path='C:/Users/Mr.Goldss/Desktop/switch back to risk management/return to work program/'
    filename=filename
    df=pd.read_excel(path+filename+'.xlsx',sheet_name='Data',skiprows=4)
    df=df[df['Claim Number'].notna()]
    df=df[['Claim Number','Financial Category','Amount']]
    return df
df_trans=read_transaction_database('3 10 2020 TTD TPall dept after 2016D')
df=df_trans.copy()
#%%
#df_trans.dropna(how='any',inplace=True)
df_trans['Financial Category']=df_trans['Financial Category'].str.strip()
df_trans=df_trans[df_trans['Financial Category'].isin(['Temp Total Disability','Temp Partial Disability'])]
df_trans=pd.DataFrame(df_trans.groupby(['Claim Number','Financial Category'])['Amount'].sum()).reset_index()
df_trans=pd.DataFrame(df_trans.pivot(index='Claim Number', columns='Financial Category', values='Amount'))
df_trans.fillna(0, inplace=True)
df_trans=df_trans.reset_index()
df_trans['Claim Number']=df_trans['Claim Number'].str.strip()
# %%
'''Step 2: get claim basic info like cause \,gender, dept,etc.'''
#filter: loss date >=2016/1/1

def read_claim_basic_info(filename):
    path='C:/Users/Mr.Goldss/Desktop/switch back to risk management/return to work program/'
    filename=filename
    df=pd.read_excel(path+filename+'.xlsx',sheet_name='Data',skiprows=4)
    df=df[df['Claim Number'].notna()]
    column_needed=['Claim Number','Loss Date','Total Incurred (Incurred)','Department (MC)','Division',\
        'Coverage','Cause','Body Parts','Wage Rate','Occupation','Gender','Marital Status','OSHA Recordable']
    df=df[column_needed]
    return df
df_basic=read_claim_basic_info('3 10 2021 claim basic info')
df_basic['Claim Number']=df_basic['Claim Number'].str.strip()

# %%
'''Step 3: get loss days info'''
def read_loss_days_info(filename):
    path='C:/Users/Mr.Goldss/Desktop/switch back to risk management/return to work program/'
    filename=filename
    df=pd.read_excel(path+filename+'.xlsx',sheet_name='Data',skiprows=4)
    df=df[df['Claim Number'].notna()]
    column_needed=['Claim Number','Primary Claimant','Is Accommodated',\
        'Transitional Duty','Indemnity Paid','All Lost Days','All Restr. Days']
    df=df[column_needed]
    return df
df_loss_days=read_loss_days_info('3 10 2021 loss days info')
df_loss_days=df_loss_days.groupby('Claim Number')[['Primary Claimant','Is Accommodated',\
        'Transitional Duty','Indemnity Paid','All Lost Days','All Restr. Days']].first()
df_loss_days=df_loss_days.reset_index()        
df_loss_days['Claim Number']=df_loss_days['Claim Number'].str.strip()``
`# %%
'''Step 4: combine everything together(base on df_loss_days)'''
df_loss_days1=df_trans.merge(df_basic,how='left',on='Claim Number')
df_loss_days2=df_loss_days1.merge(df_loss_days,how='left',on='Claim Number')
df_loss_days_final=df_loss_days2.copy()
'''Step 5: take care of missing data'''
# %%

# %%
#problem 1, I found 489 records come with TTD and TPD
    #around 50% of them come with missing value in Is Accomodated and Transitional Duty Column
    #We need a solution to fill those missing values, for example inpute with False or go back
    #to manipulate them one by one

#problem 2,there are 26 record come with TTD and TPD but missing basic value like lost days, restr days and Indemnity, why? 
 #reason: missing in the original database
 #Solution 2: we are missing 26 loss days infomation, find each of them and fill those value in Origami Risk
loss_days_final=df_loss_days_final[df_loss_days_final['All Lost Days'].notna()]
print(loss_days_final.info())
p2=df_loss_days_final[df_loss_days_final['All Lost Days'].isna()]
print(p2.info())

loss_days_final.to_excel("RTW combined data clean.xlsx")
p2.to_excel("RTW lost days, restr days and Indemnity info.xlsx")
#%%
#problem 3, TTD+TPD != Indemnity, why?

df_loss_days_final['diff']=df_loss_days_final['Indemnity Paid']-df_loss_days_final['Temp Partial Disability']-df_loss_days_final['Temp Total Disability']
#print(df_loss_days_final.info())
p3=df_loss_days_final[df_loss_days_final['diff']>0]
# %%
