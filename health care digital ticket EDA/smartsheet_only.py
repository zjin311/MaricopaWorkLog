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
# %%
def smt_data_extraction():
        
    xMytoken='a0rcziibs8bzzchmpqwgyusd4n' 
    SHEET_ID='8485507110856580'
    xSheet = smartsheet.Smartsheet(xMytoken)

    tt = xSheet.Sheets.get_sheet(SHEET_ID) 
    def simple_sheet_to_dataframe(sheet):
        col_names = [col.title for col in sheet.columns]
        rows = []
        for row in sheet.rows:
            cells = []
            for cell in row.cells:
                cells.append(cell.value)
            rows.append(cells)
        data_frame = pd.DataFrame(rows, columns=col_names)
        return data_frame
    sm_data=simple_sheet_to_dataframe(tt)


    datt_only=datt[:10]
    path='C:/Users/Mr.Goldss/Desktop/beltmann inventory(smartsheet)/data/'
    filename='MARICOPA COUNTY COMPLETE INVENTORY TRANSACTIONS'
    need_col=['PROJECT','CATEGORY','ITEM/ SKU','DESCRIPTION','SOURCE','SHIPPER'
    ,'CASE COUNT','# INNER CASE CTN',	'# PER INNER CTN','TOTAL QTY',
    'RECEIVED DATE',	'BALANCE',	'CUSTOMER NOTES','qty out','full pallet out',	'RELEASE DATE','Workorder Nbr','Release to cust:']
    df_smart=sm_data
    df_smart=df_smart[need_col]
    return df_smart
df_smart=smt_data_extraction()
# %%
#def smart_preprocessing(df_smart):
df_smart.columns=df_smart.columns.str.lower()
df=df_smart
df=df.dropna(subset=['balance','qty out'],how='all')
#df=df[(df['balance']!=0) & (df['qty out']!=0)]


# normal col df1
col_to_fill=['project', 'category', 'item/ sku', 'description', 'source', 'shipper',
        'case count', '# inner case ctn', '# per inner ctn','received date', 'workorder nbr']
df1=df[col_to_fill]

df1.fillna(method='ffill',inplace=True)

#special col df2
sepcial_col=['customer notes','full pallet out','qty out','total qty','balance',\
    'release date','release to cust:']
df2=df[sepcial_col]
df2['full pallet out']=np.where(df2['full pallet out']==True,1,0)
df2['full pallet out']=df2['full pallet out'].astype('int64')
df2['qty out'].fillna(0,inplace=True)
df2['total qty'].fillna(method='ffill',inplace=True)
df2=df2[df2['customer notes']!='Vault 12 week']
df_smt=pd.concat([df1, df2], axis=1)
df_smt.loc[df_smt['balance']=='#UNPARSEABLE','balance']=0
df_smt['category']=df_smt['category'].str.lower()
df_smt['release date']=df_smt['release date'].replace('M-07162020-8','2020-07-16')
df_smt['release date']=df_smt['release date'].replace('M-10232020-4','2020-10-23')
df_smt['release date']=df_smt['release date'].replace('FACE MASK Taiwan cloth','2020-08-27')
df_smt['release date']=df_smt['release date'].replace('Hand sanitizer 33 oz per bottle','2020-08-27')

df_smt=df_smt[df_smt['release to cust:'].notna()]
df_smt['release date'].fillna(method='ffill',inplace=True)
#%%
#df_smt=df_smt[~df_smt['release date'].isin(['x','X'])]
df_smt['release date']=pd.to_datetime(df_smt['release date'],errors='coerce')
df_smt['release date'].fillna(method='ffill',inplace=True)
df_smt['month']=pd.DatetimeIndex(df_smt['release date']).month
replace_product_type={'medical or surgical masks':'surgical mask','alternative gowns':'alt. gowns','kn95 masks':'kn95',\
    'cleaning/disinfectant':'sanitizer','eye protection':'goggles','face shield':'face shield','gloves':'gloves',\
        'n95 masks':'n95','hand sanitizer/hygiene':'sanitizer'}

df_smt.category=df_smt.category.replace(replace_product_type)
#return df_smt
#df_smt=smart_preprocessing(df_smart)
#%%

'''
def create_facility_type_for_smartsheet():
    


    facility_type_df=pd.read_excel('C:/Users/Mr.Goldss/Desktop/health care digital ticket EDA/Facility Type for hermann.xlsx',\
        sheet_name='MARICOPA DAILY RELEASE REPORT')
    facility_type_df=facility_type_df[['Facility type','Release to cust:']]

    facility_type_df2=pd.DataFrame(facility_type_df.groupby('Release to cust:')['Facility type'].first())
    facility_type_df2=facility_type_df2.reset_index()


    dept=facility_type_df2['Release to cust:'].str.lower()
    dept=dept.str.strip()
    facility_type=facility_type_df2['Facility type'].str.lower()
    facility_type=facility_type.str.strip()
    facility_type_list=list(facility_type.unique())


    def create_dict_from_col(facility_type_df2):
        
        dept=facility_type_df2['Release to cust:'].str.lower()
        dept=dept.str.strip()
        facility_type=facility_type_df2['Facility type'].str.lower()
        facility_type=facility_type.str.strip()
        facility_type_list=list(facility_type.unique())
        return dict(zip(dept, facility_type)) 
    facility_type_list=list(facility_type.unique())
    facility_type_dict=create_dict_from_col(facility_type_df2)

    #%%
    update_facility_type_df=pd.read_excel('C:/Users/Mr.Goldss/Desktop/health care digital ticket EDA/Updated Facility Type List .xlsx')
    update_facility_type_df.columns=['Release to cust:','Facility type']
    update_facility_type_dict=create_dict_from_col(update_facility_type_df)

    def Merge(dict1, dict2):
        res = {**dict1, **dict2}
        return res
    full_facility_type_dict=Merge(facility_type_dict,update_facility_type_dict)
    #%%
    df_smt['release to cust:']=df_smt['release to cust:'].str.lower()
    df_smt['facility type']=df_smt['release to cust:'].replace(full_facility_type_dict)
    df_smt['facility type'].fillna('long term care/assisted care/group home/skilled nursing/hospice',inplace=True)

    uniq_type=set(full_facility_type_dict.values())
    uniq_type=list(uniq_type)
    df_smt['facility type']=df_smt['facility type'].apply(lambda x:'long term care/assisted care/group home/skilled nursing/hospice' if x not in uniq_type else x)
    df_smt['facility type']=df_smt['facility type'].replace({'long term/skilled nursing/rehabilitation':'long term care/assisted care/group home/skilled nursing/hospice',\
        'Fqhc/community health center':'fqhc/community health center','Fqhc/community center':'fqhc/community health center','Home health':'home health',\
            'Hospice care':'hospice care'})
    return df_smt

df_smt=create_facility_type_for_smartsheet()
'''
#%%



