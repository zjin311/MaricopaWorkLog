
#%%

import smartsheet, csv
import pandas as pd
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

#===============================================================



#%%
import datetime 
import numpy as np
import gspread
import matplotlib
import seaborn as sns
import matplotlib.pyplot as plt
import string
import os

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
datt=str(datetime.datetime.now())
datt_only=datt[:10]
path='C:/Users/Mr.Goldss/Desktop/beltmann inventory(smartsheet)/data/'
filename='MARICOPA COUNTY COMPLETE INVENTORY TRANSACTIONS'
need_col=['PROJECT','CATEGORY','ITEM/ SKU','DESCRIPTION','SOURCE','SHIPPER'
,'CASE COUNT','# INNER CASE CTN',	'# PER INNER CTN','TOTAL QTY',
'RECEIVED DATE',	'BALANCE',	'CUSTOMER NOTES','qty out','full pallet out',	'RELEASE DATE','Workorder Nbr','Release to cust:']
df_smart=sm_data
df_smart=df_smart[need_col]


#%%
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

#%%
df_smt=smart_preprocessing(df_smart)
#%%
df_smt['release date']=df_smt['release date'].replace('M-07162020-8','2020-07-16')
df_smt['release date']=df_smt['release date'].replace('M-10232020-4','2020-10-23')
df_smt['release date']=df_smt['release date'].replace('FACE MASK Taiwan cloth','2020-08-27')
df_smt['release date']=df_smt['release date'].replace('Hand sanitizer 33 oz per bottle','2020-08-27')
df_smt['release date']=pd.to_datetime(df_smt['release date'])

df_smt['month']=pd.DatetimeIndex(df_smt['release date']).month
df_smt.description=df_smt.description.str.lower()
#%%

#%%
def gloves_burn_rate_monthly(df):
    df=df[df['category']=='gloves']
    
    return df.groupby('month')['qty out'].sum()



    #total_gloves=df['qty out'].sum()
    #num_of_month=df['month'].max()-df['month'].min()
    #gloves_burn_rate=total_gloves/num_of_month
    #return gloves_burn_rate

smt_gloves_giveout_monthly=gloves_burn_rate_monthly(df_smt)
result=pd.merge(dircks_gloves_after_june,smt_gloves_giveout_monthly,\
    left_on=dircks_gloves_after_june.index,right_on=smt_gloves_giveout_monthly.index,how='outer')
result.fillna(0,inplace=True)

result.columns=['month','dircks gloves giveout','beltmann gloves giveout']
result['total giveout']=result['dircks gloves giveout']+result['beltmann gloves giveout']
gloves_avg_giveout=result['total giveout'].sum()/(result.month.max()-result.month.min())
result.to_csv('gloves total giveout from Jun to OCT.csv')
#%%


#we only care about top product type


'''
cat_chart_save_path='C:/Users/Mr.Goldss/Desktop/beltmann inventory(smartsheet)/category chart update daily/'
for i in df_smt.category.unique():


    if sum(df_smt['category']==i)>10:
        try:


            fig, ax1 = plt.subplots()
            fig.set_size_inches(25, 12)
            sns_plot =sns.lineplot(x='release date',y='qty out',hue='category',\
                data=df_smt[df_smt['category']==i],ci=True, markers=True)
            plt.xticks(fontsize=18,rotation=90)
            plt.xlabel('release date', fontsize=20)
            plt.ylabel('Out Amount', fontsize=25)
            plt.title('Beltmann total give out {}'.format(i), fontsize=30)
            plt.tight_layout() 
            plt.show()
            fig.savefig(cat_chart_save_path+'beltmann inventory {}{}.png'.format(i,datt_only)) 
        
        except:
            print('bye bye ')
            continue


'''
#%%
'''
#Line Chart: Total amount given by product type

cat_chart_save_path='C:/Users/Mr.Goldss/Desktop/beltmann inventory(smartsheet)/category chart update daily/'

target_cat=['Medical or Surgical Masks',  'Gloves','N95 Masks','Gowns']
fig, ax1 = plt.subplots()
fig.set_size_inches(25, 12)
sns_plot =sns.lineplot(x='release date',y='qty out',hue='category',\
    data=df_smt[df_smt['category'].isin(target_cat)],ci=True,estimator='sum')
plt.xticks(fontsize=18,rotation=90)
plt.yticks(fontsize=28)
plt.xlabel('release date', fontsize=20)
plt.ylabel('Out Amount', fontsize=25)
plt.title('Beltmann total give out {}'.format(datt_only), fontsize=30)
plt.legend(fontsize=20)
plt.tight_layout() 
plt.show()

fig.savefig(cat_chart_save_path+'beltmann give out amount by category.png') 

'''

#%%

'''daily update 1 :==> category by balance'''

#cond_still_in_storage=df_smt['full pallet out']==0

df_smt.loc[df_smt['balance']=='#UNPARSEABLE','balance']=0
df_smt['category']=df_smt['category'].str.lower()




def handle_72000_missing_date(df):
    df_missing=df[df['qty out']==72000]
    df_no_missing=df[df['qty out']!=72000]
    df_missing['release date']=df_missing['release date'].fillna('2020-08-03')
    dddd=pd.concat([df_missing,df_no_missing])
    return dddd
df_smt=handle_72000_missing_date(df_smt)



df_smt['release date'].fillna(method='ffill',inplace=True)
df_smt=df_smt[df_smt['release to cust:'].notna()]
#%%
df_smt['release to cust:']=df_smt['release to cust:'].str.lower()
df_smt['release to cust:']=df_smt['release to cust:'].replace({'circle city':'circle the city'})

#%%
                        '''create facility type for smartsheet'''


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


#%%

def create_facility_type_smt(df_smt):
    df_smt['facility type']=df_smt['release to cust:'].replace(facility_type_dict)
    df_smt['facility type'].fillna('long term care/assisted care/group home/skilled nursing/hospice',inplace=True)
    df_smt['facility type']=df_smt['facility type'].apply(lambda x:'long term care/assisted care/group home/skilled nursing/hospice' if x not in facility_type_list else x)
    return df_smt
df_smt=create_facility_type_smt(df_smt)
facility_type_group=df_smt.groupby(['facility type','category','month'])['qty out'].sum()
#ddf=pd.pivot_table(df_smt,values='qty out',index='facility type',columns='month',aggfunc=np.sum)
smt_facility_type_group=pd.DataFrame(facility_type_group.reset_index())

replace_product_type={'medical or surgical masks':'surgical mask','alternative gowns':'alt. gowns','kn95 masks':'kn95',
'cleaning/disinfectant':'sanitizer','eye protection':'goggles','face shield':'face shield','gloves':'gloves',
'n95 masks':'n95','hand sanitizer/hygiene':'sanitizer'}

smt_facility_type_group.category=smt_facility_type_group.category.replace(replace_product_type)

smt_facility_type_group['facility type']=smt_facility_type_group['facility type'].replace('long term/skilled nursing/rehabilitation','long term care/assisted care/group home/skilled nursing/hospice')







#%%
'''concat dirck facility type dataset with smt '''
Both_facility_group_combined=pd.concat([smt_facility_type_group,dircks_facility_type_group])
Both_facility_group_combined=Both_facility_group_combined.groupby(['facility type','category','month'])['qty out'].sum()
Both_facility_group_combined=pd.DataFrame(Both_facility_group_combined.reset_index())
#%%
Both_facility_group_combined.to_csv(str(datt[:10])+' beltmann&dircks ppe giveout by facility type.csv')
#%%

'''
#search for 5 specific facilities
#df_smt[df_smt['release to cust:']=='Southwellness, LLC.']
search_nae=['southwellness','circle the city','csrv consulting',]
#df_smt[df_smt['release to cust:'].str.contains('valle')]

tttt=df_smt[df_smt['release to cust:'].isin(search_nae)]
tttt=tttt.groupby(['release to cust:','facility type','category','release date'])['qty out'].sum()
tttt=pd.DataFrame(tttt).reset_index()

needed_facility_total_giveout=pd.concat([tttt1,tttt])
needed_facility_total_giveout['facility type']=needed_facility_total_giveout['facility type'].replace(\
    {'Ltcf':'long term care/assisted care/group home/skilled nursing/hospice'})
needed_facility_total_giveout.to_csv('found 4 facility record  No Sonora Added date.csv')





'''

#%%

#replace smartsheet category with dirks standard or the other way around
cat_out=df_smt[df_smt['qty out']>0][['category','qty out','release date']]
beltmann_cat=cat_out.category.unique()


print(beltmann_cat)
replace_cat={'medical or surgical masks':'surgical mask','alternative gowns':'alt. gowns','kn95 masks':'kn95',
'cleaning/disinfectant':'sanitizer','eye protection':'goggles','face shield':'face shield','gloves':'gloves',
'n95 masks':'n95','hand sanitizer/hygiene':'sanitizer'}

cat_out.category=cat_out.category.replace(replace_cat)



df_both=pd.concat([cat_out,df_dir_concat])

df_both['release date']=pd.to_datetime(df_both['release date'])

df_both['week_of_year']=df_both['release date'].dt.strftime('%U')
df_both['month']=df_both['release date'].dt.strftime('%m')
df_both['month']=df_both['month'].astype('int')
df_both=df_both.sort_values(by='release date')
df_both['qty out']=df_both['qty out'].astype(int)
#%%

df_both.to_excel(' r total giveout March to today.xlsx')
























#%%
'''project past 90 days total giveout by category and facility Type'''
def date_filter(df,month):
    return df[df.month>=month]

date_filter(df_both,8)

#%%

'''cumsum total give out'''

pivot = pd.pivot_table(df_both, values="qty out", index=["week_of_year"], columns=["category"], aggfunc=np.sum)
pivot = pivot.cumsum()
pivot=pivot[['surgical mask',  'n95', 'gowns', 'gloves']]


fig, ax1 = plt.subplots()
fig.set_size_inches(25, 12)
pivot.plot(ax=ax1,ylim=(0,2000000))
plt.xticks(fontsize=18,rotation=90)
plt.yticks(fontsize=28)
plt.xlabel('week_of_year', fontsize=20)
plt.ylabel('Out Amount', fontsize=25)
plt.title('Cumsum total give out until {}'.format(datt_only), fontsize=30)
plt.legend(fontsize=20)

plt.tight_layout()
save_fig('cumsum total give out')
plt.show()   

   
#%%


cat_chart_save_path='C:/Users/Mr.Goldss/Desktop/health care digital ticket EDA/images/give out analysis/'

target_cat=['surgical mask', 'alt. gowns', 'kn95', 'sanitizer', 'goggles',
       'face shield', 'n95', 'gowns', 'gloves','powered air-purifying respirator (papr)']
for i in target_cat:
    fig, ax1 = plt.subplots()
    fig.set_size_inches(25, 12)
    sns_plot =sns.lineplot(x='week_of_year',y='qty out',\
        data=df_both[df_both.category==i],ci=False,estimator='sum')
    plt.xticks(fontsize=18,rotation=90)
    plt.yticks(fontsize=28)
    plt.xlabel('week_of_year', fontsize=20)
    plt.ylabel('Out Amount', fontsize=25)
    plt.title('Beltmann total give out {} {}'.format(i,datt_only), fontsize=30)
    plt.legend(fontsize=20)
    plt.tight_layout() 
    fig.savefig(cat_chart_save_path+'Total amount'+i+'.png') 
    plt.show()

fig.savefig(cat_chart_save_path+'beltmann give out amount by category.png') 



#%%

'''total given both inventory data '''
df_total_out_both_alltime=df_both.groupby('category')['qty out'].sum().sort_values(ascending=False)
df_total_out_both_alltime.to_csv('Covid-19 PPE total giveout.csv')
#%%










#all time high
type_unique=df_both.category.unique()
maxnumber=[]
for i in type_unique:
    maxnumber.append(df_both[df_both['category']==i].\
        groupby(['week_of_year','category'])['qty out'].agg('sum').sort_values(ascending=False)[0])

print(maxnumber)


#%%
#create dictionary
alltimehighbytype=list(zip(type_unique,maxnumber))

print(alltimehighbytype)
#%%

df_max=pd.DataFrame(alltimehighbytype,columns=['category','All_Time_High_Weekly'])







#%%
#how many week?
how_many_week_tile=int(dt.datetime.now().strftime('%U'))-int(min(df_both['release date']).strftime('%U'))



#%%
df_ana=df_both.groupby(['category']).sum()/how_many_week_tile
df_ana=df_ana.reset_index().sort_values(by='qty out',ascending=False)
df_ana.columns=['category','weekly qty mean']
df_ana['weekly qty mean']=df_ana['weekly qty mean'].round()
#type_unique=df_both.category.unique()
#%%


'''std'''
std_by_type=[]
for i in type_unique:
    std_by_type.append(df_both[df_both['category']==i].groupby('week_of_year')['qty out'].agg(np.std).mean())

df_ana['std']=std_by_type
df_ana['std']=df_ana['std'].fillna(0)
df_ana['std']=df_ana['std'].round()

#%%
# 95
df_ana=df_ana.merge(df_max,on='category')
#%%
df_ana['weekly 95 confident order amount']=df_ana['std']*2+df_ana['weekly qty mean']
df_ana['weekly 99.7 confident order amount']=df_ana['std']*3+df_ana['weekly qty mean']
df_ana['weekly 4 std confident order amount']=df_ana['std']*4+df_ana['weekly qty mean']
df_ana['weekly 5 std confident order amount']=df_ana['std']*5+df_ana['weekly qty mean']
df_ana['weekly 6 std confident order amount']=df_ana['std']*6+df_ana['weekly qty mean']
#%%
#10 week total amount to order
df_ana['10 week order (95 confident)']=df_ana['weekly 95 confident order amount']*10
df_ana['10 week order (99.7 confident)']=df_ana['weekly 99.7 confident order amount']*10
df_ana['10 week order (4std confident)']=df_ana['weekly 4 std confident order amount']*10
df_ana['10 week order (4std confident)']=df_ana['weekly 4 std confident order amount']*10
df_ana['10 week order (5std confident)']=df_ana['weekly 5 std confident order amount']*10
df_ana['10 week order (6std confident)']=df_ana['weekly 6 std confident order amount']*10


#%%
df_ana.to_csv('8-10-2020 PPE Order amount prediction with confidence level by zixiang michael Jin.csv')


























#%%
#& cond2_check_andy
smart_stock=df_smt[cond_still_in_storage ].groupby('category')['balance'].sum().sort_values(ascending=False)
smart_stock=smart_stock.reset_index()
smart_stock=pd.DataFrame(smart_stock)
smart_stock['Date']=datt_only
smart_stock
beltmann_inventory=pd.read_csv('C:/Users/Mr.Goldss/Desktop/beltmann inventory(smartsheet)/daily inventory level report/beltmann daily category balance.csv')

beltmann_inventory=pd.concat([beltmann_inventory, smart_stock], ignore_index=True)
beltmann_inventory['Date']=pd.to_datetime(beltmann_inventory['Date'])
#%%
beltmann_inventory.to_csv('C:/Users/Mr.Goldss/Desktop/beltmann inventory(smartsheet)/daily inventory level report/'+'{}beltmann daily category balance.csv'.format(datt_only))

#%%

datetime_object = datetime.datetime.now()

fig, ax1, = plt.subplots()
fig.set_size_inches(25, 12)
sns.barplot(x='category',y='balance',data=smart_stock)
plt.xticks(fontsize=18,rotation=90)
plt.xlabel('category', fontsize=15)
plt.ylabel('balance in Million', fontsize=20)
plt.title('Beltmann Inventory Distribution', fontsize=30)
for index, value in enumerate(smart_stock.balance):
    plt.text(index,value+400, str(value),fontsize=12)
plt.tight_layout() 

punctuations = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''

no_punct = ""
for char in datt:
   if char not in punctuations:
       no_punct = no_punct + char

save_path='C:/Users/Mr.Goldss/Desktop/beltmann inventory(smartsheet)/image/'
plt.savefig(save_path+'beltmann inventory {}.png'.format(no_punct)) 

# %%
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

#,'Matthew.Melendez@maricopa.gov'
email_list2=['andrew.weflen@maricopa.gov','zjin24@asu.edu','richard.langevin@maricopa.gov','gerry.gill@maricopa.gov','Ashley.Miranda@maricopa.gov']
email_list=['zjin24@asu.edu']

for i in email_list2:

    send_email_attach(
        from_who='zixiangjin921@gmail.com',
    to_who=i,subject_name='Beltmann Dailt inventory update',
    content='Inventory data exclude Vault 12 week. data from smartsheet, update date {}'.format(datt),
    attach_file_name='beltmann inventory.png',
    file_path=save_path+'beltmann inventory '+no_punct+'.png',
    password='Intelcorei7')

# %%
