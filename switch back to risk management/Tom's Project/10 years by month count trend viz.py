#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime 
#from datetime import datetime
import missingno as msno
import smartsheet
import os
import gspread
from functools import partial
datt=str(datetime.datetime.now())[:10]



#%%
'''temporary project'''
tem_path='C:/Users/Mr.Goldss/Desktop/switch back to risk management/chirs request/'
df_basicinfo=df=pd.read_excel(tem_path+'2-1-2021-employee basic info.xlsx')


df_basicinfo=df_basicinfo[df_basicinfo['Cause']=='SLIP/TRIP/FALL']

chris_columns=['Claim Number','EmployeeNumber','Status','Hire Date','Birth Date','Occupation','Gender','Coverage','Cause','Body Parts','Total Incurred (Incurred)','Department (MC)']

df_basicinfo2=df_basicinfo[chris_columns]
#%%
df_basicinfo2.to_excel(tem_path+'SLIP_TRIP_FALL accidents in CY2020.xlsx')




#%%
df=pd.read_excel('1_26_2020 Jinclaimbasicin2021012616378d302c4844ba96585ab734f85742.xlsx',skiprows=4)
# %%
PROJECT_ROOT_DIR = "."
CHAPTER_ID = "10 years trending by month"
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID)
os.makedirs(IMAGES_PATH, exist_ok=True)


def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)

#%%
df_n=df[['Claim Number','Coverage','Loss Date','Department (MC)']]
df_n['month']=df_n['Loss Date'].dt.strftime('%m')
df_n.dropna(how='any',inplace=True)
wanted_coverage=['Workers Compensation','General Liability','Auto Physical Damage']
df_m=df_n[df_n['Coverage'].isin(wanted_coverage)]
df_d=df_m.groupby(['month','Coverage']).size()
df_d=pd.DataFrame(df_d).reset_index()
df_d.columns=['month','coverage','count']

# %%

def lineplot_by_year(year):

    fig, ax1, = plt.subplots()
    fig.set_size_inches(25, 12)
    sns.lineplot(x='year_month',y='count',hue='coverage',\
        data=df_d[df_d['year']==year],ax=ax1,\
            ci=False, markers=True)
    plt.xticks(fontsize=18,rotation=90)
    plt.yticks(fontsize=18)
    plt.xlabel('Year_Month', fontsize=30)
    plt.ylabel('Count', fontsize=25)
    plt.title('{} Monthly Trend'.format(year), fontsize=30)
    plt.legend(loc=2, prop={'size': 20})
    plt.tight_layout() 
    save_fig(str(year)+' Monthly trending')
    plt.show()

# %%
years=list(df_d.year.unique())
for i in years:
    lineplot_by_year(i)
    print('\n')
    print('======')

# %%
def barplot_by_year():
    
    fig, ax1, = plt.subplots()
    fig.set_size_inches(60, 25)
    sns.barplot(x='month',y='count',hue='coverage',\
        data=df_d,ax=ax1)
    plt.xticks(fontsize=60)
    plt.yticks(fontsize=60)
    plt.xlabel('Month', fontsize=80)
    plt.ylabel('Count', fontsize=80)
    plt.title('Monthly Trend', fontsize=60)
    plt.legend(loc=2, prop={'size': 40})
    plt.tight_layout() 
    save_fig(' 10 years Monthly(sum) trending by category hist')
    plt.show()


# %%
barplot_by_year()

#%%

df_noHue=pd.DataFrame(df_d.groupby('month')['count'].sum()).reset_index()
#%%
def barplot_by_year_noHue():
    
    fig, ax1, = plt.subplots()
    fig.set_size_inches(60, 25)
    sns.barplot(x='month',y='count',\
        data=df_noHue,ax=ax1)
    plt.xticks(fontsize=60)
    plt.yticks(fontsize=60)
    plt.xlabel('Month', fontsize=80)
    plt.ylabel('Count', fontsize=80)
    plt.title('Monthly Trend', fontsize=60)
    plt.legend(loc=2, prop={'size': 40})
    plt.tight_layout() 
    save_fig(' 10 years Monthly(sum) trending hist')
    plt.show()

barplot_by_year_noHue()
#%%


cov=list(df_d.coverage.unique())
print(cov)
'''
for i in cov:
    barplot_by_year(i)
    print('\n')
    print('======')
'''

# %%

barplot_by_year('General Liability')
# %%

barplot_by_year('Workers Compensation')
# %%
