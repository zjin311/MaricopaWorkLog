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
def send_email_attach(from_who,to_who,subject_name,content,attach_file_name,file_path,password):
    
    import smtplib 
    from email.mime.multipart import MIMEMultipart 
    from email.mime.text import MIMEText 
    from email.mime.base import MIMEBase 
    from email import encoders 
   
    fromaddr = from_who
    toaddr = to_who
   
    msg = MIMEMultipart() 
 
    msg['From'] = fromaddr 
 
    msg['To'] = toaddr 
  

    msg['Subject'] = subject_name
  

    body = content
  

    msg.attach(MIMEText(body, 'plain')) 

    filename = attach_file_name
    attachment = open(file_path, "rb") 
  

    p = MIMEBase('application', 'octet-stream') 

    p.set_payload((attachment).read()) 
 
    encoders.encode_base64(p) 
   
    p.add_header('Content-Disposition', "attachment; filename= %s" % filename) 
  

    msg.attach(p) 
  

    s = smtplib.SMTP('smtp.gmail.com', 587) 

    s.starttls() 
  
    s.login(fromaddr, password) 

    text = msg.as_string() 
  
    s.sendmail(fromaddr, toaddr, text) 
    attachment.close() 
email_list2=['andrew.weflen@maricopa.gov','zjin24@asu.edu','richard.langevin@maricopa.gov','gerry.gill@maricopa.gov','Ashley.Miranda@maricopa.gov']
email_list=['zjin24@asu.edu']

#%%
#def save image routine
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
#%%

#-----------------------------------------------------------

            #Dircks Data extract, cleaning
            #Dircks Data extract, cleaning

#-----------------------------------------------------------


path='C:/Users/Mr.Goldss/Desktop/health care digital ticket EDA/'
filename='4-8-2021 Dircks Updated Facility Itemized Tracker'
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
#%%
df_clean['Delivered Date']=pd.to_datetime(df_clean['Delivered Date'], errors = 'coerce')

df_clean['Delivered Date'].fillna('2020-12-03',inplace=True)
df_clean['Delivered Date']=pd.to_datetime(df_clean['Delivered Date'], errors = 'coerce')

#%%
def del_wrong_ems(df_clean):

    del_cond1=(df_clean['Delivered Date']>='2020-04-01') & (df_clean['Delivered Date']<='2020-04-30')
    del_cond2=(df_clean['Facility Type'] =='EMS') & (df_clean['Facility Name'] !='Phoenix Fire Resource Management')
    to_be_drop_EMS=df_clean.loc[del_cond1 & del_cond2]
    Ems_drop_list=to_be_drop_EMS.index
    df_clean_copy=df_clean
    df_clean_EMS=df_clean_copy.drop(Ems_drop_list)
    return df_clean_EMS
df_clean_EMS=del_wrong_ems(df_clean)

#%%


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
    df_clean_EMS['Facility Name']=df_clean_EMS['Facility Name'].str.lower()
    type_replacement={'disposable\xa0masks':'surgical mask','masks':'surgical mask'}
    df_clean_EMS['Type']=df_clean_EMS['Type'].replace(type_replacement)

    return df_clean_EMS
deep_clean_str_manipulate(df_clean_EMS)


def dircks_replace_facility_type_match_smt(df_clean_EMS):
    
    #dircks_facility_type=df_clean_EMS['Facility Type'].unique()
    facility_type_replace={'Ems':'ems/fire/law enforcement','Ltcf':'long term care/assisted care/group home/skilled nursing/hospice',
    'County':'county','Private practice':'private practice','Fqhc':'fqhc/community health center','Acute':'hospital',
    'Juvenile':'ems/fire/law enforcement','Urgent care':'hospital','Tribal':'jurisdiction','School':'school',
    'Private':'private practice','Other':'other','Home healthcare':'home health','Government':'jurisdiction',
    'Juvenile probations':'ems/fire/law enforcement'}
    df_clean_EMS['Facility Type']=df_clean_EMS['Facility Type'].replace(facility_type_replace)
    return df_clean_EMS
df_clean_EMS=dircks_replace_facility_type_match_smt(df_clean_EMS)


#%%

                # the line below is used for data concat#
              # used for Total Amount/ Daily amount analysis#
df_dir_concat=df_clean_EMS[['Type','Total Amount','Delivered Date']]
df_dir_concat.columns=['category','qty out','release date']



                # those line below is used for data concat#
              # used for facility type+type+month+total amount analysis#
dircks_facility_type_group=df_clean_EMS.groupby(['Facility Type','Type','Month'])['Total Amount'].sum()

dircks_facility_type_group=pd.DataFrame(dircks_facility_type_group.reset_index())
dircks_facility_type_group.columns=['facility type','category','month','qty out']
dircks_facility_type_group['facility type']=dircks_facility_type_group['facility type'].replace({'long term/skilled nursing/rehabilitation':'long term care/assisted care/group home/skilled nursing/hospice',\
        'Fqhc/community health center':'fqhc/community health center','Fqhc/community center':'fqhc/community health center','Home health':'home health',\
            'Hospice care':'hospice care'})
# %%


#-----------------------------------------------------------

                #Smartsheet Data extract, cleaning
                #Smartsheet Data extract, cleaning
#-----------------------------------------------------------

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
#%%



                # the line below is used for data concat#
              # used for Total Amount/ Daily amount analysis#

cat_out=df_smt[df_smt['qty out']>0][['category','qty out','release date']]
                # those line below is used for data concat#
              # used for facility type+type+month+total amount analysis#

facility_type_group=df_smt.groupby(['facility type','category','month'])['qty out'].sum()

smt_facility_type_group=pd.DataFrame(facility_type_group.reset_index())





# %%

'''donw with cleaning and ready for concat and more analysis'''

#report section
#report 1:  Total All time
os.chdir(path)


df_both=pd.concat([cat_out,df_dir_concat])
df_both['year']=df_both['release date'].dt.year.astype('int')
df_both['month']=df_both['release date'].dt.month.astype('int')
df_both['week']=df_both['release date'].dt.week.astype('int')
#df_both['week']=np.where(df_both['year']==2021,df_both['week']+52,df_both['week'])
#weekly burn rate
#%%
df_both['category']=df_both['category'].str.strip()
df_both['category']=df_both['category'].replace(\
    {'ventilator ':'ventilator','test':'testing supplies','t':'testing supplies',\
        'hand sanitizer':'sanitizer','disinfectant':'sanitizer','thermometers':'thermometer',\
            'gowns':'gown','other items':'other','needles':'needle','mask':'surgical mask',\
                'ventilators':'ventilator supplies','ventilator':'ventilator supplies',\
                    'testing suppliesing supplies':'testing supplies','glasses':'goggles',\
                        'ventilator':'ventilator supplies','wipes':'disinfectants','sanitizer':'disinfectants'})

#df_both.to_csv('Tableau viz Total PPE2.csv')
#%%
burn_rate_df=df_both.groupby(['year','month'])['qty out'].sum().reset_index()
burn_rate_df=burn_rate_df.sort_values(by=['year','month'],ascending=True)
#burn_rate_df=burn_rate_df.iloc[:12,:]
burn_rate_df.to_csv(datt+'monthly total given amount.csv')

#%%


#%%
for i in email_list2:
    send_email_attach(
    from_who='zixiangjin921@gmail.com',
    to_who=i,
    subject_name='Monthly Total Given',
    content='Report 5.  update time: '+datt+' total give out per month',
    attach_file_name=datt+'monthly total given amount.csv',
    file_path='C:/Users/Mr.Goldss/Desktop/health care digital ticket EDA/'+datt+'monthly total given amount.csv',
    password='numerodata123!')


#burn_rate_df=burn_rate_df[(burn_rate_df.year==2021)&burn_rate_df.month.isin([4,7,12])]
#%%
burn_rate_df.plot()
#%%


fig, ax = plt.subplots(figsize=(15, 9))
palette = sns.color_palette("ch:2.5,-.2,dark=.3", 10)
sns.regplot(x='month', y="qty out", data=burn_rate_df,ax=ax)
ax.set_title('Total Request by week', fontsize = 20, loc='center', fontdict=dict(weight='bold'))
ax.set_xlabel('Number of week', fontsize = 16, fontdict=dict(weight='bold'))
ax.set_ylabel('Applied Facility', fontsize = 16, fontdict=dict(weight='bold'))
plt.show()



#%%


'''continue here'''

df_total_out_both_alltime=df_both.groupby('category')['qty out'].sum().sort_values(ascending=False)
#df_total_out_both_alltime.to_csv(datt+' Covid-19 PPE total giveout.csv')
#%%

# Email yout report out
for i in email_list2:
    send_email_attach(
    from_who='zixiangjin921@gmail.com',
    to_who=i,
    subject_name='Total giveout update',
    content='Report 1.  update time: '+datt+' total give out from both warehouse all time',
    attach_file_name=datt+' Covid-19 PPE total giveout.csv',
    file_path='C:/Users/Mr.Goldss/Desktop/health care digital ticket EDA/'+datt+' Covid-19 PPE total giveout.csv',
    password='numerodata123!')


# %%
#report 2: Total today ot Total any day
dat='2021-04-07'
def total_today(dat):

    total_by_date=df_both[df_both['release date']==dat]
    total_by_date=total_by_date.groupby('category')['qty out'].sum().sort_values(ascending=False)
    return total_by_date
total_today=total_today(dat)
total_today.to_csv(dat+' daily Covid-19 PPE total giveout.csv')
#%%
for i in email_list2:
    send_email_attach(
    from_who='zixiangjin921@gmail.com',
    to_who=i,
    subject_name='Daily Total giveout update',
    content='Report 2.  update time: '+dat+' daily Covid-19 PPE total giveout.csv',
    attach_file_name=datt+' Daily Covid-19 PPE total giveout.csv',
    file_path='C:/Users/Mr.Goldss/Desktop/health care digital ticket EDA/'+dat+' daily Covid-19 PPE total giveout.csv',
    password='numerodata123!')






# %%
#report 3: Total All time by month and category
Both_facility_group_combined=pd.concat([smt_facility_type_group,dircks_facility_type_group])
Both_facility_group_combined=Both_facility_group_combined.groupby(['facility type','category','month'])['qty out'].sum()
Both_facility_group_combined=pd.DataFrame(Both_facility_group_combined.reset_index())
Both_facility_group_combined.to_csv(datt+' details about beltmann&dircks ppe giveout by facility type.csv')
#%%
for i in email_list2:
    send_email_attach(
    from_who='zixiangjin921@gmail.com',
    to_who=i,
    subject_name='Total giveout update by facility type and month in details',
    content='Report 3.  update time: '+datt+' details about beltmann&dircks ppe giveout by facility type and month.csv',
    attach_file_name=datt+' Daily Covid-19 PPE total giveout.csv',
    file_path='C:/Users/Mr.Goldss/Desktop/health care digital ticket EDA/'+datt+' details about beltmann&dircks ppe giveout by facility type.csv',
    password='numerodata123!')


#%%

facility_type_summary=Both_facility_group_combined.groupby('facility type')['qty out'].sum().sort_values(ascending=False)
facility_type_summary.to_csv(datt+' facility type summary.csv')
#%%
for i in email_list2:
    send_email_attach(
    from_who='zixiangjin921@gmail.com',
    to_who=i,
    subject_name='Facility type summary',
    content='Report 4.  update time: '+datt+' Summary about beltmann&dircks ppe giveout by facility type.csv',
    attach_file_name=datt+' Daily Covid-19 PPE total giveout.csv',
    file_path='C:/Users/Mr.Goldss/Desktop/health care digital ticket EDA/'+datt+' facility type summary.csv',
    password='numerodata123!')


# %%



# %%
copydf=df_both
# %%
df_both=pd.DataFrame(df_both.groupby(['week','category'])['qty out'].sum()).reset_index()
#df_both=pd.DataFrame(df_both).reset_index()
# %%
type_unique=df_both.category.unique()
# %%
maxnumber=[]
for i in type_unique:
    maxnumber.append(df_both[df_both['category']==i]['qty out'].max())

# %%
alltimehighbytype=list(zip(type_unique,maxnumber))

print(alltimehighbytype)

df_ana=pd.DataFrame(alltimehighbytype,columns=['Type','All_Time_High_Weekly'])

# %%
mean_by_type=[]
for i in type_unique:
    mean_by_type.append(int(df_both[df_both['category']==i]['qty out'].mean()))

all_time_mean_by_type=list(zip(mean_by_type,type_unique))


#add to df

df_ana['weekly_mean']=mean_by_type
print(all_time_mean_by_type)
# %%
std_by_type=[]
for i in type_unique:
    std_by_type.append(df_both[df_both['category']==i]['qty out'].agg(np.std))

df_ana['std']=std_by_type
df_ana['std']=df_ana['std'].fillna(0)
df_ana['std']=df_ana['std'].round()
# %%

df_ana['95 confident order amount']=df_ana['std']*2+df_ana['weekly_mean']
df_ana['99 confident order amount']=df_ana['std']*3+df_ana['weekly_mean']
df_ana['4std amount']=df_ana['std']*4+df_ana['weekly_mean']
df_ana['5std amount']=df_ana['std']*5+df_ana['weekly_mean']
df_ana['6std amount']=df_ana['std']*6+df_ana['weekly_mean']

df_ana['10 week order (95 confident)']=df_ana['95 confident order amount']*10
df_ana['10 week order (99 confident)']=df_ana['99 confident order amount']*10
df_ana['10 week order (4std confident)']=df_ana['4std amount']*10


df_ana['12 week order (95 confident)']=df_ana['95 confident order amount']*12
df_ana['12 week order (99 confident)']=df_ana['99 confident order amount']*12
df_ana['12 week order (4std confident)']=df_ana['4std amount']*12

# %%
df_ana.to_excel('2-2-2021 ppe six-sigma.xlsx')
# %%
