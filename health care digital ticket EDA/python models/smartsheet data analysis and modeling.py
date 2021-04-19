#%%
import datetime
import pandas as pd
import gspread
import matplotlib
import seaborn as sns
import matplotlib.pyplot as plt
import string
#%%
path='C:/Users/Mr.Goldss/Desktop/beltmann inventory(smartsheet)/data/'
filename='MARICOPA COUNTY COMPLETE INVENTORY TRANSACTIONS'
need_col=['PROJECT','CATEGORY','ITEM/ SKU','DESCRIPTION','SOURCE','SHIPPER'
,'CASE COUNT','# INNER CASE CTN',	'# PER INNER CTN','TOTAL QTY',
'SERVICE DATE',	'BALANCE',	'CUSTOMER NOTES','qty out','full pallet out',	'RELEASE DATE','Workorder Nbr']
df_smart=pd.read_excel(path+filename+'.xlsx')
df_smart=df_smart[need_col]


#%%
def smart_preprocessing(df_smart):
       df_smart.columns=df_smart.columns.str.lower()
       df=df_smart
       df=df.dropna(subset=['balance'])


# normal col df1
       col_to_fill=['project', 'category', 'item/ sku', 'description', 'source', 'shipper',
              'case count', '# inner case ctn', '# per inner ctn',
              'service date','release date', 'workorder nbr']
       df1=df[col_to_fill]

       df1.fillna(method='ffill',inplace=True)

#special col df2
       sepcial_col=['customer notes','full pallet out','qty out','total qty','balance']
       df2=df[sepcial_col]
       df2['full pallet out'].fillna(0,inplace=True)
       df2['full pallet out']=df2['full pallet out'].astype('int64')
       df2['qty out'].fillna(0,inplace=True)
       df2['total qty'].fillna(method='ffill',inplace=True)
       df_smt=pd.concat([df1, df2], axis=1)
       return df_smt
#%%


df_smt=smart_preprocessing(df_smart)


#%%

'''daily update 1 :==> category by balance'''

cond_still_in_storage=df_smt['full pallet out']==0

cond2_check_andy=df_smt['service date']<='9/8/2020'


smart_stock=df_smt[cond_still_in_storage & cond2_check_andy].groupby('category')['balance'].sum().sort_values(ascending=False)


smart_stock.reset_index()

smart_stock=pd.DataFrame(smart_stock)


#%%

datetime_object = datetime.datetime.now()

fig, ax1, = plt.subplots()
fig.set_size_inches(25, 12)
sns.barplot(x=smart_stock.index,y='balance',data=smart_stock)
plt.xticks(fontsize=18,rotation=90)
plt.xlabel('category', fontsize=15)
plt.ylabel('balance in Million', fontsize=20)
plt.title('Beltmann Inventory Distribution', fontsize=30)
for index, value in enumerate(smart_stock.balance):
    plt.text(index,value+400, str(value),fontsize=12)
plt.tight_layout() 

punctuations = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
datt=str(datetime.datetime.now())
no_punct = ""
for char in datt:
   if char not in punctuations:
       no_punct = no_punct + char

save_path='C:/Users/Mr.Goldss/Desktop/beltmann inventory(smartsheet)/image/'
plt.savefig(save_path+'beltmann inventory {}.png'.format(no_punct)) 



#smart_stock.to_csv('Andy and smartsheet.csv')

 # %%


# %%
