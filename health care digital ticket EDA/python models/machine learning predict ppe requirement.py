#%%
import smartsheet, csv
import pandas as pd
import datetime 
import numpy as np
import gspread
import matplotlib
import seaborn as sns
import matplotlib.pyplot as plt
import string
import os


#%%
'''step one: gather smt data'''
def setp_1_smt_data():

    '''
    Set up the data connection with smartsheet
    '''
    xMytoken='a0rcziibs8bzzchmpqwgyusd4n' 
    SHEET_ID='8485507110856580'
    # Smartsheet Token
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


    PROJECT_ROOT_DIR = "."
    CHAPTER_ID = "give out analysis"
    IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID)
    os.makedirs(IMAGES_PATH, exist_ok=True)

    def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
        path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
        print("Saving figure", fig_id)
        if tight_layout:
            plt.tight_layout()
        plt.savefig(path, format=fig_extension, dpi=resolution)


    datt=str(datetime.datetime.now())
    datt_only=datt[:10]
    path='C:/Users/Mr.Goldss/Desktop/beltmann inventory(smartsheet)/data/'
    filename='MARICOPA COUNTY COMPLETE INVENTORY TRANSACTIONS'
    need_col=['PROJECT','CATEGORY','ITEM/ SKU','DESCRIPTION','SOURCE','SHIPPER'
    ,'CASE COUNT','# INNER CASE CTN',	'# PER INNER CTN','TOTAL QTY',
    'RECEIVED DATE',	'BALANCE',	'CUSTOMER NOTES','qty out','full pallet out',	'RELEASE DATE','Workorder Nbr','Release to cust:']
    df_smart=sm_data
    df_smart=df_smart[need_col]
    


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
    


    #handle release date column.

    df_smt['release date']=df_smt['release date'].replace('M-07162020-8','2020-07-16')
    df_smt['release date']=df_smt['release date'].replace('M-10232020-4','2020-10-23')
    df_smt['release date']=df_smt['release date'].replace('FACE MASK Taiwan cloth','2020-08-27')
    df_smt['release date']=df_smt['release date'].replace('Hand sanitizer 33 oz per bottle','2020-08-27')
    df_smt=df_smt[df_smt['release to cust:'].notna()]
    df_smt['release date'].fillna(method='ffill',inplace=True)
    df_smt=df_smt[~df_smt['release date'].isin(['x','X'])]
    df_smt['release date']=pd.to_datetime(df['release date'])

    def handle_72000_missing_date(df):

        df_missing=df[df['qty out']==72000]
        df_no_missing=df[df['qty out']!=72000]
        df_missing['release date']=df_missing['release date'].fillna('2020-08-03')
        dddd=pd.concat([df_missing,df_no_missing])
        return dddd
    df_smt=handle_72000_missing_date(df_smt)




    df_smt=df_smt[df_smt['release to cust:'].notna()]


    df_smt['month']=pd.DatetimeIndex(df_smt['release date']).month
    df_smt.description=df_smt.description.str.lower()

        
    df_smt.loc[df_smt['balance']=='#UNPARSEABLE','balance']=0
    df_smt['category']=df_smt['category'].str.lower()


    # create facility type
    df_smt['release to cust:']=df_smt['release to cust:'].str.lower()
    df_smt['release to cust:']=df_smt['release to cust:'].replace({'circle city':'circle the city'})


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
    df_smt['facility type']=df_smt['release to cust:'].replace(full_facility_type_dict)
    df_smt['facility type'].fillna('long term care/assisted care/group home/skilled nursing/hospice',inplace=True)

    uniq_type=set(full_facility_type_dict.values())
    uniq_type=list(uniq_type)
    df_smt['facility type']=df_smt['facility type'].apply(lambda x:'long term care/assisted care/group home/skilled nursing/hospice' if x not in uniq_type else x)

    
    df_smt['facility type']=df_smt['facility type'].replace('long term/skilled nursing/rehabilitation','long term care/assisted care/group home/skilled nursing/hospice')

    replace_product_type={'medical or surgical masks':'surgical mask','alternative gowns':'alt. gowns','kn95 masks':'kn95',
    'cleaning/disinfectant':'sanitizer','eye protection':'goggles','face shield':'face shield','gloves':'gloves',
    'n95 masks':'n95','hand sanitizer/hygiene':'sanitizer'}

    df_smt.category=df_smt.category.replace(replace_product_type)


    


    return df_smt[['category','qty out','facility type','release date','month']]
df_smt=setp_1_smt_data()
# %%
'''step two: gather dircks data'''
def setp_2_dircks_data():
    path='C:/Users/Mr.Goldss/Desktop/health care digital ticket EDA/'
    filename='10 1 2020 Facility Itemized Tracker (1)'
    df=pd.read_excel(path+filename+'.xlsx',sheet_name='New ALL')


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
    df_clean_EMS=deep_clean_str_manipulate(df_clean_EMS)

    def dircks_replace_facility_type_match_smt(df_clean_EMS):

        facility_type_replace={'Ems':'ems/fire/law enforcement','Ltcf':'long term care/assisted care/group home/skilled nursing/hospice',
        'County':'county','Private practice':'private practice','Fqhc':'fqhc/community health center','Acute':'hospital',
        'Juvenile':'ems/fire/law enforcement','Urgent care':'hospital','Tribal':'jurisdiction','School':'school',
        'Private':'private practice','Other':'other','Home healthcare':'home health','Government':'jurisdiction',
        'Juvenile probations':'ems/fire/law enforcement'}
        df_clean_EMS['Facility Type']=df_clean_EMS['Facility Type'].replace(facility_type_replace)

        return df_clean_EMS['Facility Type']
    df_clean_EMS['Facility Type']=dircks_replace_facility_type_match_smt(df_clean_EMS)
    df_clean_EMS.columns=['release date','facility name','facility type','Item Description',
       'Product Source','category','Units', 'QTY Per UOM', 'qty out',
       'size', 'week_of_year', 'month']
    return df_clean_EMS[['category','qty out','facility type','release date','month']]
df_clean_EMS=setp_2_dircks_data()

# %%
'''step 3: concat step one & step two '''
df_ml_both=pd.concat([df_smt,df_clean_EMS])


#%%
ddft=pd.DataFrame(df_ml_both.groupby(['month','category','facility type'])['qty out'].sum()).reset_index()
ddft.to_excel('Total giveout by month and facility type details.xlsx')
ddft_all_giveout=ddft.groupby('category')['qty out'].sum().sort_values(ascending=False)
ddft_all_giveout.to_excel('Total giveout summary.xlsx')
#ddft2=ddft[ddft['category']=='gloves']
#ddft2.to_excel('3--11 gloves giveout by facility type.xlsx')
#%%
criticalPPE=['gloves','gowns','n95','surgical mask']
ddft=ddft[ddft.category.isin(criticalPPE)]
df_ml_both=ddft
# %%









'''step 4: Prepare data for ML (missing/group/encoding)'''
#only on facility type missing
#def ML_preprocessing():
'''
def create_day_year_drop_release_date(df_ml_both):
    df_ml_both['facility type']=df_ml_both['facility type'].fillna('long term care/assisted care/group home/skilled nursing/hospice')
    df_ml_both['release date']=pd.to_datetime(df_ml_both['release date'])
    df_ml_both['year'] = df_ml_both['release date'].dt.strftime('%Y')
    df_ml_both['day'] = df_ml_both['release date'].dt.strftime('%d')
    df_ml_both.drop('release date', axis=1,inplace=True)
    return df_ml_both
df_ml_both=create_day_year_drop_release_date(df_ml_both)
'''

#%%
X,y=df_ml_both[['category','facility type', 'month']],df_ml_both['qty out']

X["category"]=X["category"].astype('category')
X["facility type"]=X["facility type"].astype('category')
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=2,test_size=0.3)

def simple_imputer(X_train):

    X_train["category"]= X_train["category"].cat.codes
    X_train["facility type"]= X_train["facility type"].cat.codes
    return X_train
X_train=simple_imputer(X_train)

#%%
'''step 5--1:try decision tree predict'''

from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

bag_clf = BaggingClassifier(
    DecisionTreeClassifier(random_state=42), n_estimators=50,
    max_samples=100, bootstrap=True, random_state=42)
bag_clf.fit(X_train, y_train)


X_test=simple_imputer(X_test)
y_pred = bag_clf.predict(X_test)

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, y_pred))

# NO!!!!!!!!!!!!!!!


# %%
'''step 5--2:try Linear regression '''
from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)

y_pred=lin_reg.predict(X_test)

from sklearn.metrics import mean_squared_error

y_pred = lin_reg.predict(X_test)
lin_mse = mean_squared_error(y_test, y_pred)
lin_rmse = np.sqrt(lin_mse)
lin_rmse


# %%
'''step 5--3:try DecisionTreeRegressor '''
from sklearn.tree import DecisionTreeRegressor

tree_reg = DecisionTreeRegressor(random_state=42)
tree_reg.fit(X_train, y_train)

y_pred = tree_reg.predict(X_test)
tree_mse = mean_squared_error(y_test, y_pred)
tree_rmse = np.sqrt(tree_mse)
tree_rmse
# %%
def release_date_manipulate(df):
    df['release date']=df['release date'].replace('M-07162020-8','2020-07-16')
    df['release date']=df['release date'].replace('M-10232020-4','2020-10-23')
    df['release date']=df['release date'].replace('FACE MASK Taiwan cloth','2020-08-27')
    df['release date']=df['release date'].replace('Hand sanitizer 33 oz per bottle','2020-08-27')

    df=df[df['release to cust:'].notna()]
    df['release date'].fillna(method='ffill',inplace=True)
    df=df[~df['release date'].isin(['x','X'])]
    df['release date']=pd.to_datetime(df['release date'])
    
    return df
df_smt=release_date_manipulate(df_smt)
