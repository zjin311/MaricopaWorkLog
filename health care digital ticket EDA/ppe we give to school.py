#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime 
import missingno as msno
import smartsheet
import os
import gspread
datt=str(datetime.datetime.now())
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


#,'Matthew.Melendez@maricopa.gov'
email_list2=['andrew.weflen@maricopa.gov','zjin24@asu.edu','richard.langevin@maricopa.gov','gerry.gill@maricopa.gov','Ashley.Miranda@maricopa.gov']
email_list=['zjin24@asu.edu']

#%%
'''step one : school supply from dircks latest update at  8-18-2019'''


path='C:/Users/Mr.Goldss/Desktop/health care digital ticket EDA/'
filename='8-18-2020 Facility Itemized Tracker (1)'
df=pd.read_excel(path+filename+'.xlsx',sheet_name='New ALL')
def beltmann():
        
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

dircks_ppe=df_clean_EMS.copy()
dircks_school=dircks_ppe[dircks_ppe['Facility Type']=='School']
dircks_school=dircks_school[['Facility Name','Delivered Date','Type','Total Amount']]
dircks_school.columns=['Facility Name','Delivered Date','Category','Total Amount']


# %%
'''step two :  '''

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

#%%

datt_only=datt[:10]
path='C:/Users/Mr.Goldss/Desktop/beltmann inventory(smartsheet)/data/'
filename='MARICOPA COUNTY COMPLETE INVENTORY TRANSACTIONS'
need_col=['PROJECT','CATEGORY','ITEM/ SKU','DESCRIPTION','SOURCE','SHIPPER'
,'CASE COUNT','# INNER CASE CTN',	'# PER INNER CTN','TOTAL QTY',
'RECEIVED DATE',	'BALANCE',	'CUSTOMER NOTES','qty out','full pallet out',	'RELEASE DATE','Workorder Nbr','Release to cust:']
df_smart=sm_data
df_smart=df_smart[need_col]

def smart_preprocessing(df_smart):
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
       
       #df_smt['full pallet out'] not in []
       return df_smt
df_smt=smart_preprocessing(df_smart)
#%%
df_smt['release date']=df_smt['release date'].replace('M-07162020-8','2020-07-16')
df_smt['release date']=df_smt['release date'].replace('FACE MASK Taiwan cloth','2020-08-27')
df_smt['release date']=df_smt['release date'].replace('Hand sanitizer 33 oz per bottle','2020-08-27')
df_smt['release date']=pd.to_datetime(df_smt['release date'])
df_smt.loc[df_smt['balance']=='#UNPARSEABLE','balance']=0
df_smt['category']=df_smt['category'].str.lower()
#%%
'''
df_smt['release date'].fillna(method='ffill',inplace=True)
df_smt=df_smt[df_smt['release to cust:'].notna()]
'''
df_smt['release to cust:']=df_smt['release to cust:'].str.lower()
df_smt['release to cust:']=df_smt['release to cust:'].fillna('unknown')



# %%
school_list=['2 schools- Interntaional Commerce and Humanities and Sciences',
'A Challenge Foundation Academy',
'Academia Del Pueblo',
'ACCEL',
'ACCLAIM Charter.',
'AIBT',
'All Aboard. Charter School',
'Arcadia High School',
'Arizona Knowledge Empowerment',
'arizona Lutheran Academy',
'AWAKENING SEED SCHOOL',
'Az Aspire Academy',
'Candeo Schools',
'canyon elementary',
'CASS Vista Colina',
'Cesar Chavez Foundation',
'Christ Greenfield Lutheran',
'Creighton School District',
'Dayspring Kindergarten',
'DEER VALLEY UNIFIED SCHOOL',
'desert choice schools',
'emmaus lutheran',
'Ethos Academy',
'Fowler Elementary School District',
'Grace Christian Academy',
'Hi Star Center School',
'imagine schools rosefield',
'Integrity Education Incorporated',
'International School of Arizona',
'Isaac School District',
'Kyrene School District',
'laveen elementary school district',
'Metropolitan Arts Institute.',
'Painted Rock Academy',
'Paradise Honors Schools',
'Pendergast Elementary School District',
'Phoenix Elementary S.D.',
'Premier School',
'Saddle Mountain Unified School District',
'Saints Simon and Jude School',
'San Tan Montessori',
'San Tan Montessori School',
'sierra linda high school',
'Step Up Schools',
'Success School',
'Tempe Elementary District #3',
'TESD #17',
'Tolleson Union High School',
'UNION ELEMENTARY SCHOOL DIST.',

'Phoenix Elementary S.D.',
'sierra linda high school',
'Tolleson Union High School',
'Getz Elementary School',
'Glendale Elementary School District',
'Tolleson Union High School',
'Desert Jewish Academy',
'Getz Elementary School',
'Tolleson Union High School',
'Desert Jewish Academy',
'Basis Chandler Primary North',
'Gilbert Public Schools',
'Twenty First Century Charte',
'Buckeye Elementary School District #33',
'BASIS Ahwatukee',
'Deer Valley Unified School District',
'Ebony House/Elba House',
'Glendale Elementary School District #40'


]
# %%
school_list=list(map(lambda x: x.lower(),school_list))
# %%
df_smt_school_only=df_smt[df_smt['release to cust:'].isin(school_list)]
df_smt_school_only=df_smt_school_only[['release to cust:','release date','category','qty out']]

df_smt_school_only.columns=['Facility Name','Delivered Date','Category','Total Amount']

# %%
both_school=pd.concat([df_smt_school_only,dircks_school])
both_school=both_school.sort_values(by='Delivered Date',ascending=False)
both_school.reset_index()
both_school.to_csv('total school give out details.csv')
# %%

replace_category_name={'medical or surgical masks':'surgical mask',\
    'n95 masks':'n95'}

both_school.Category=both_school.Category.replace(replace_category_name)

both_school_cat=both_school.groupby('Category')['Total Amount'].sum().sort_values(ascending=False)

both_school_cat.to_csv('total school give out from both beltmann & dircks.csv')
# %%
