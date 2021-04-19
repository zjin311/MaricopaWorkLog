
#%%  Read data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime 
from datetime import datetime
import missingno as msno

#%config InlineBackend.figure_format = 'retina'
#%matplotlib inline
#%%
path='C:/Users/Mr.Goldss/Desktop/health care digital ticket EDA/'
filename='11-24 -2020 Facility Itemized Tracker (003)'
df=pd.read_excel(path+filename+'.xlsx',sheet_name='New ALL')

df.shape

#check the missing partner:

#msno.matrix(df)
#%%
#msno.bar(df)
#%%
#msno.heatmap(df)

#%%                     fix basic formatting: preprocessing
'''data preprocessing'''

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

#%%

#%%
'''group by facility type'''
#df_clean_EMS.groupby('Facility Type')['Total Amount'].sum()
#%%
#df_clean_EMS[df_clean_EMS['Facility Type']=='Other']

df_clean_EMS['Facility Name']=df_clean_EMS['Facility Name'].str.lower()

tttt1=df_clean_EMS[df_clean_EMS['Facility Name'].str.contains('valle del sol')]

tttt1=tttt1.groupby(['Facility Name','Facility Type','Type','Delivered Date'])['Total Amount'].sum()
tttt1=pd.DataFrame(tttt1).reset_index()
tttt1.columns=['release to cust:','facility type','category','release date','qty out']
#%%
                                                '''concat'''

#%%

df_dir_concat=df_clean_EMS[['Type','Total Amount','Delivered Date']]
df_dir_concat.columns=['category','qty out','release date']

thomas_replace_cat={'Medical or Surgical Masks':'surgical mask', 'Alternative Gowns':'alt. gowns', 'Eye Protection':'eye protection',
       'Reusable cloth face covering':'reusable masks', 'N95 Masks':'n95', 'Face Shield':'face shield',
       'Hand Sanitizer/Hygiene':'hand sanitizer', 'Gowns':'gowns', 'Other Items':'other', 'Coveralls':'coveralls',
       'Gloves':'gloves','Cleaning/Disinfectant':'disinfectant'}
df_dir_concat.category=df_dir_concat.category.replace(thomas_replace_cat)












#%%
'''product resource analysis'''
source_to_replace={'AZ':'State Purchased','Maricopa':'Maricopa Purchased','AZ/MasCache':'State Purchased',\
    'Missing':'SNS','Maricoap':'Maricopa Purchased','Purchased/Maricopa':'Maricopa Purchased'}

df_clean_EMS['Product Source']=df_clean_EMS['Product Source'].replace(source_to_replace)

resource_df=df_clean_EMS.groupby('Product Source')['Total Amount'].sum().sort_values(ascending=False)
resource_df=resource_df.reset_index()



#%%
'''facility type changing func'''
def dircks_replace_facility_type_match_smt(df_clean_EMS):

    dircks_facility_type=df_clean_EMS['Facility Type'].unique()
    facility_type_replace={'Ems':'ems/fire/law enforcement','Ltcf':'long term care/assisted care/group home/skilled nursing/hospice',
    'County':'county','Private practice':'private practice','Fqhc':'fqhc/community health center','Acute':'hospital',
    'Juvenile':'ems/fire/law enforcement','Urgent care':'hospital','Tribal':'jurisdiction','School':'school',
    'Private':'private practice','Other':'other','Home healthcare':'home health','Government':'jurisdiction',
    'Juvenile probations':'ems/fire/law enforcement'}
    df_clean_EMS['Facility Type']=df_clean_EMS['Facility Type'].replace(facility_type_replace)

    dircks_facility_type_group=df_clean_EMS.groupby(['Facility Type','Type','Month'])['Total Amount'].sum()

    dircks_facility_type_group=pd.DataFrame(dircks_facility_type_group.reset_index())
    dircks_facility_type_group.columns=['facility type','category','month','qty out']
    return dircks_facility_type_group
dircks_facility_type_group=dircks_replace_facility_type_match_smt(df_clean_EMS)



#%%





#%%
def dircks_gloves_monthly(df):
    df2=df[df['Type']=='gloves']
    df2=df2[df2.Month>=6]
    #total_gloves=df['Total Amount'].sum()
    #num_of_month=df['Month'].max()-df['Month'].min()
    #gloves_burn_rate=total_gloves/num_of_month
    return df2

dircks_gloves_after_june=dircks_gloves_monthly(df_clean_EMS)

dircks_gloves_after_june=dircks_gloves_after_june.groupby('Month')['Total Amount'].sum()
#%%
df_clean_EMS['ticket_unique_id']=list(map(str,df_clean_EMS['Delivered Date']))+df_clean_EMS['Facility Name']







#%%
'''school'''

school=df_clean_EMS[df_clean_EMS['Facility Type']=='School']





#%%

#python profiling
'''
df_clean_report=df_clean_EMS[['Delivered Date', 'Facility Name', 'Facility Type',
       'Product Source', 'Type', 'Total Amount',
       'size', 'week_of_year', 'Month']]
from pandas_profiling import ProfileReport
profile = ProfileReport(df_clean_report, title='Pandas Profiling Report', explorative=True)
profile.to_file(path+"your_report.html")
                                               
'''

#%%                    
'''analysis product type by size'''
def what_perc_to_order(t):
    total=sum(df_clean_EMS[df_clean_EMS['Type']==str(t)].groupby(['Type','size'])['Total Amount'].sum())
    perc1=df_clean_EMS[df_clean_EMS['Type']==str(t)].groupby(['Type','size'])['Total Amount'].apply(lambda x:x.sum()/total)
    return 'total give out {type:} amount is {amount:}'.format(type=t,amount=total) ,  perc1
unique_type=list(df_clean_EMS['Type'].unique())
def find_type_size_prec(unique_type):
    

    for i in unique_type:
        
        a=what_perc_to_order(i)
        print(a)
        print('--------------------------------------')
        print()

find_type_size_prec(unique_type)   
#you can also just cakk what perc to order func for individual product type
# like this ||
#           VV     
#what_perc_to_order('gloves')








#%%
'''analysis product type by amount within a time range'''
def time_range_type_total(smalldate,bigdate):

    c1=df_clean_EMS['Delivered Date']>=smalldate
    c2=df_clean_EMS['Delivered Date']<=bigdate
    TimerangeTotal=df_clean_EMS[c1 & c2].groupby('Type')['Total Amount'].sum()
    return TimerangeTotal.sort_values(ascending=False)
time_range_type_total('2020-01-11','2020-12-25')



#%%
'''analysis reusable gowns''' 

df_reusable=df_clean_EMS[df_clean_EMS['Item Description'].str.contains('reusable')==True]

#%%
df_disposable=df_clean_EMS[df_clean_EMS['Item Description'].str.contains(r'disposable\.mask')==True]




#%%
'''analysis All Total'''
new_total=df_clean_EMS.groupby('Type')['Total Amount'].sum()
new_total.sort_values(ascending=False)
#new_total.to_csv('{}total.csv'.format(date.today()))
# %%


''' analysis: average PPE supply each batch exclude 'Acute' '''


def supply_per_batch():
    num_unique_batch=df_clean_EMS[df_clean_EMS['Facility Type']!='Acute'].groupby(['Delivered Date','Facility Name'])['Total Amount'].sum().shape[0]

    batch_avg=pd.DataFrame(df_clean_EMS[df_clean_EMS['Facility Type']!='Acute'].groupby(['Type'])['Total Amount'].sum()/num_unique_batch)

    return batch_avg['Total Amount'].apply(int).sort_values(ascending=False)

print(supply_per_batch())
#%%

'''analysis: how many ppe we give per ticket by different facility type'''
def avg_ppe_by_facility_type(df_clean_EMS):


    unique_faclity_type=df_clean_EMS['Facility Type'].unique()
    facility_type_count=[]

    for i in range(len(unique_faclity_type)):
        #with shape func to calculate count of unique facility in different facility type
        facility_type_count.append(df_clean_EMS[df_clean_EMS['Facility Type']==unique_faclity_type[i]].groupby(['Delivered Date','Facility Name'])['Total Amount'].sum().shape[0])
    
    
    unique_facility_type_count=dict(zip(unique_faclity_type,facility_type_count))

    for key,value in unique_facility_type_count.items():
        print(key)

        print(df_clean_EMS[df_clean_EMS['Facility Type']==key].groupby(['Type'])['Total Amount'].sum()/value)
        print('-----------------')
        print()

avg_ppe_by_facility_type(df_clean_EMS)


# %%
'''how many gloves we give out from Jun 4 to Jun 25 on average

'''

def avg_product_giveout(date1,date2):
    cond1=(df_clean_EMS['Delivered Date']>=date1)
    cond2=(df_clean_EMS['Delivered Date']<=date2)
    cond3=(df_clean_EMS['Facility Type']=='Ltcf')
    temp_df=df_clean_EMS[cond1 & cond2 &cond3 ].groupby('Type')['Total Amount'].sum()
    return temp_df
temp_df=pd.DataFrame(avg_product_giveout('2019-07-01' , '2020-06-30'))
#temp_df.sort_values(by=['Type','Total Amount'],ascending=False,inplace=True )
#temp_df.to_csv('_Jun-4 to -25 product give out average by size.csv')



#%%
sum_ems_by_type=df_clean_EMS[df_clean_EMS['Facility Type']=='Ems'].groupby(['Type'])['Total Amount'].sum().sort_values(ascending=False)
sum_ems_by_type=pd.DataFrame(sum_ems_by_type).reset_index()
print(sum_ems_by_type)
sum_ems_by_type.to_csv('7_29 Gerry asked EMS spply info.csv')

# %%
# Email :Daily total amount by type send to Email
data_version=str(df_clean_EMS['Delivered Date'].max())
data_version=data_version[:10]

new_total=df_clean_EMS.groupby('Type')['Total Amount'].sum()
new_total=new_total.sort_values(ascending=False)
punctuations = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
datt=str(dt.datetime.now())
no_punct = ""
for char in datt:
   if char not in punctuations:
       no_punct = no_punct + char
savedname='{}total.csv'.format(no_punct[:9])

dir_to_save='C:/Users/Mr.Goldss/Desktop/health care digital ticket EDA/type and total amount/'
new_total.to_csv(dir_to_save+savedname)

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
email_list2=['andrew.weflen@maricopa.gov',\
    'eoc.logistics@maricopa.gov','zjin24@asu.edu','richard.langevin@maricopa.gov','gerry.gill@maricopa.gov','Matthew.Melendez@maricopa.gov','Ashley.Miranda@maricopa.gov']
email_list=['zjin24@asu.edu','eoc.logistics@maricopa.gov']

for i in email_list:

    send_email_attach(
        from_who='zixiangjin921@gmail.com',
    to_who=i,subject_name='Customer received PPE Amount',
    content="Confirmed delivered PPE supply by type. \
        1 week data delay. Data version:{}. \
            Data team didn't received any ticket this week".format(data_version),
    attach_file_name='confirmed total amount--Jin.csv',
    file_path=dir_to_save+savedname,
    password='Intelcorei7')

#%%
send_email_attach(
    from_who='zixiangjin921@gmail.com',
to_who=i,subject_name='EMS received PPE Amount',
content="EMS supply",
attach_file_name='confirmed EMS total amount--Jin.csv',
file_path='C:/Users/Mr.Goldss/Desktop/python module and function/7_29 Gerry asked EMS spply info.csv',
password='Intelcorei7')

























#%%
     '''Viz'''
          '''Viz'''
               '''Viz'''
                    '''Viz'''
                         '''Viz'''
                              '''Viz'''
                                   '''Viz'''
#%%

fig, ax1, = plt.subplots()
fig.set_size_inches(25, 12)
sns.countplot(x='Facility Type',data=df_clean_EMS,ax=ax1)
plt.xticks(fontsize=18,rotation=90)
plt.xlabel('Facility', fontsize=15)
plt.ylabel('count', fontsize=20)
plt.title('Beltmann Inventory Distribution', fontsize=30)

plt.tight_layout() 
plt.show()

# %%
'''sns.lineplot---->date as x axis'''
#we only care about 5 product type
type_needed=['n95','surgical mask','gowns','gloves']
fig, ax1, = plt.subplots()
fig.set_size_inches(25, 12)
sns.lineplot(x='Month',y='Total Amount',hue='Type',\
    data=df_clean_EMS[df_clean_EMS['Type'].isin(type_needed)],ax=ax1,\
        ci=False, markers=True, style='Type')
plt.xticks(fontsize=18,rotation=90)
plt.xlabel('Month', fontsize=20)
plt.ylabel('Amount', fontsize=25)
plt.title('day', fontsize=30)

plt.tight_layout() 
plt.show()

# %%
'''sns.barplot()'''
#one thing need to realize is that the bar is not the sum but the mean

type_needed=['n95','surgical mask','gowns','gloves']
fig, ax1, = plt.subplots()
fig.set_size_inches(25, 12)
sns.barplot(x='Month',y='Total Amount',hue='Type',\
    data=df_clean_EMS[df_clean_EMS['Type'].isin(type_needed)],ax=ax1,\
        ci=False)
plt.yticks(fontsize=18)
plt.xticks(fontsize=20)
plt.xlabel('Month of the year', fontsize=20)
plt.ylabel('mean Amount', fontsize=25)
plt.title('day', fontsize=30)

plt.tight_layout() 
plt.show()
# %%
'''switch x and y and use orient=h to transpose '''
type_needed=['n95','surgical mask','gowns','gloves']
fig, ax1, = plt.subplots()
fig.set_size_inches(25, 12)
sns.barplot(y='Month',x='Total Amount',hue='Type',\
    data=df_clean_EMS[df_clean_EMS['Type'].isin(type_needed)],ax=ax1,\
        ci=False,orient='h')
plt.yticks(fontsize=18)
plt.xticks(fontsize=20)
plt.xlabel('Month of the year', fontsize=20)
plt.ylabel('mean Amount', fontsize=25)
plt.title('day', fontsize=30)

plt.tight_layout() 
plt.show()


# %%
'''how can i change the color?'''
#go to seaborn official web to find color option
# search for html color code--> find the (hex) code like this :#E315C7 
type_needed=['n95','surgical mask','gowns','gloves']
fig, ax1, = plt.subplots()
fig.set_size_inches(25, 12)
sns.barplot(y='Month',x='Total Amount',hue='Type',\
    data=df_clean_EMS[df_clean_EMS['Type'].isin(type_needed)],ax=ax1,\
        ci=False,orient='h',color='#E315C7')
plt.yticks(fontsize=18)
plt.xticks(fontsize=20)
plt.xlabel('Month of the year', fontsize=20)
plt.ylabel('mean Amount', fontsize=25)
plt.title('day', fontsize=30)

plt.tight_layout() 
plt.show()


# %%
'''hist is a good way to analysis the distribution of the data'''

fig, ax1, = plt.subplots()
fig.set_size_inches(25, 12)
sns.distplot(df_clean_EMS[(df_clean_EMS['Type']=='gloves')\
    &(df_clean_EMS['Facility Type']=='Ltcf')\
        &(df_clean_EMS['Total Amount']<=500)]['Total Amount'],ax=ax1)
plt.yticks(fontsize=18)
plt.xticks(fontsize=20)
plt.xlabel('give amount', fontsize=20)
plt.ylabel('count', fontsize=25)
plt.title('give out distribution', fontsize=30)

mean=df_clean_EMS[(df_clean_EMS['Type']=='gloves')\
    &(df_clean_EMS['Facility Type']=='Ltcf')\
        &(df_clean_EMS['Total Amount']<=500)]['Total Amount'].mean()
plt.axvline(mean, color='red')

plt.tight_layout() 
plt.show()


# %%
def draw_type_facility_type_amount_dist(Type,FacilityType,upper_level=50000):

    fig, ax1, = plt.subplots()
    fig.set_size_inches(25, 12)
    sns.distplot(df_clean_EMS[(df_clean_EMS['Type']==Type)\
        &(df_clean_EMS['Facility Type']==FacilityType)\
            &(df_clean_EMS['Total Amount']<=upper_level)]['Total Amount'],ax=ax1)
    plt.yticks(fontsize=18)
    plt.xticks(fontsize=20)
    plt.xlabel('give amount', fontsize=20)
    plt.ylabel('count', fontsize=25)
    plt.title('give out distribution', fontsize=30)

    mean=df_clean_EMS[(df_clean_EMS['Type']==Type)\
        &(df_clean_EMS['Facility Type']==FacilityType)\
            &(df_clean_EMS['Total Amount']<=upper_level)]['Total Amount'].mean()
    plt.axvline(mean, 0,1,color='red')

    plt.tight_layout() 
    plt.show()

# %%
draw_type_facility_type_amount_dist('n95','Acute',2000)

# %%
'''box plot is another good way to show distribution'''
fig, ax1, = plt.subplots()
fig.set_size_inches(25, 12)
sns.boxplot(df_clean_EMS[(df_clean_EMS['Type']=='gloves')\
    &(df_clean_EMS['Facility Type']=='Ltcf')\
        &(df_clean_EMS['Total Amount']<=5000)]['Total Amount'],ax=ax1)
plt.yticks(fontsize=18)
plt.xticks(fontsize=20)
plt.xlabel('give amount', fontsize=20)
plt.ylabel('count', fontsize=25)
plt.title('give out distribution', fontsize=30)

mean=df_clean_EMS[(df_clean_EMS['Type']=='gloves')\
    &(df_clean_EMS['Facility Type']=='Ltcf')\
        &(df_clean_EMS['Total Amount']<=500)]['Total Amount'].mean()
plt.axvline(mean, color='red')

plt.tight_layout() 
plt.show()


# %%
fig, ax1, = plt.subplots()
#fig.set_size_inches(25, 12)
sns.set(rc={'figure.figsize':(12,10)})
sns.boxplot(y='Total Amount',
            x='Month',data=df_clean_EMS,ax=ax1)
plt.tight_layout() 
plt.show()



# %%
'''sns.swarmplot(), which can show every data point with a distribution,
 is a good helper for sns.boxplot, '''


fig, ax1, = plt.subplots()
#fig.set_size_inches(25, 12)
sns.set(rc={'figure.figsize':(12,10)})
sns.boxplot(x='Total Amount',
            y='Month',data=df_clean_EMS[df_clean_EMS['Total Amount']<=5000],ax=ax1,orient='h',color='#E55C94')

sns.swarmplot(x='Total Amount',
            y='Month',data=df_clean_EMS[df_clean_EMS['Total Amount']<=5000],ax=ax1,orient='h',color='green')
plt.tight_layout() 
plt.show()


# %%
'''same as sns.violinplot'''
fig, ax1, = plt.subplots()
#fig.set_size_inches(25, 12)
#sns.set(rc={'figure.figsize':(12,10)})
sns.boxplot(x='Total Amount',
            y='Month',data=df_clean_EMS[df_clean_EMS['Total Amount']<=5000],ax=ax1,orient='h',color='#E55C94')

sns.violinplot(x='Total Amount',
            y='Month',data=df_clean_EMS[df_clean_EMS['Total Amount']<=5000],ax=ax1,orient='h',color='green')
plt.tight_layout() 
plt.show()



# %%
'''let's try sns.scatterplot()'''
fig, ax1, = plt.subplots()
#fig.set_size_inches(25, 12)
#sns.set(rc={'figure.figsize':(12,10)})
sns.scatterplot(y='Total Amount',
            x='week_of_year',data=df_clean_EMS[df_clean_EMS['Total Amount']<=5000],\
                ax=ax1,color='#E55C94')
plt.tight_layout() 
plt.show()


# %%
'''what is lmplot()==> basically: scattern plot with regression line'''
#this is not a good example, basically you need to find 2 continue variable


sns.lmplot(y='Total Amount',
            x='Month',data=df_clean_EMS[df_clean_EMS['Total Amount']<=5000],height=10)
            
plt.tight_layout() 
plt.show()

# %%
sns.set()
bin_edges=[0,500,1000,1500,2000,2500,3000,10000]

#type_to_show=df_clean_EMS['Type']=='gloves'
ax=plt.hist(df_clean_EMS['Total Amount'],bins=bin_edges,cumulative=True,density=True)
ax=plt.xlabel('name of the x axis')
ax=plt.ylabel('y')
plt.show
#%%
fig, ax, = plt.subplots()
ax=sns.swarmplot(x='Type',y='Total Amount',data=df_clean_EMS[df_clean_EMS['Type'].isin(['n95','surgical mask'])])
fig.set_size_inches(25, 12)
ax=plt.xlabel('name of the x axis')
ax=plt.ylabel('name of the y axis')
plt.show()
# %%
df_clean_EMS[df_clean_EMS['Type'].isin(['n95','surgial mask'])].Type.unique()

# %%
def ecdf(data):
    """Compute ECDF for a one-dimensional array of measurements."""
    # Number of data points: n
    n = len(data)

    # x-data for the ECDF: x
    x = np.sort(data)

    # y-data for the ECDF: y
    y = np.arange(1, n+1) / n

    return x, y

# %%
data_cond1=df_clean_EMS['Type']=='n95'
data_cond2=df_clean_EMS['Type']=='gowns'
data_cond3=df_clean_EMS['Type']=='gloves'
data_cond4=df_clean_EMS['Type']=='surgical mask'

#isin(['n95','gowns','gloves','surgical mask'])
data_condx=df_clean_EMS['Total Amount']<500
x95,y95=ecdf(df_clean_EMS[data_cond1 & data_condx]['Total Amount'])
xgowns,ygowns=ecdf(df_clean_EMS[data_cond2 & data_condx]['Total Amount'])

xglove,yglove=ecdf(df_clean_EMS[data_cond3 & data_condx]['Total Amount'])
xsur,ysur=ecdf(df_clean_EMS[data_cond4 & data_condx]['Total Amount'])


plt.plot(x95,y95,marker = '.' , linestyle = 'none')
plt.plot(xgowns,ygowns,marker = '.' , linestyle = 'none')
plt.plot(xglove,yglove,marker = '.' , linestyle = 'none')
plt.plot(xsur,ysur,marker = '.' , linestyle = 'none')

plt.legend(('n95','gowns','gloves','surgical mask'), loc='lower right')

plt.xlabel('label of x axis')
plt.ylabel('ECDF')
plt.show()

# %%
fig, ax, = plt.subplots()
fig.set_size_inches(25, 12)
data_cond_amount=df_clean_EMS[c<=5000]
ax=sns.boxplot(x='Type',y='Total Amount',data=data_cond_amount)
ax=plt.xticks(rotation=90,fontsize=24)
ax=plt.xlabel('Type')
ax=plt.ylabel('Total Amount')
plt.show()
#%%

fig, ax, = plt.subplots()
fig.set_size_inches(25, 12)
data_cond_amount=df_clean_EMS[df_clean_EMS['Total Amount']<=100000]
ax=sns.distplot(np.log(data_cond_amount['Total Amount']))
ax=plt.xticks(rotation=90,fontsize=24)
ax=plt.xlabel('Amount')
ax=plt.ylabel('y')
plt.show()




# %%
df_clean_EMS[['QTY Per UOM','Total Amount']].corr()

# %%
np.corrcoef([1,2,3,4,5],[5,4,3,2,1])

# %%
np.cov([1,2,3,4,5],[5,4,3,2,1])

# %%
df_clean_EMS['Delivered Date'].dt.dayofweek

# %%

from datetime import datetime

now = datetime.now()

# %%
#Elderly & Physically Disabled Providers Alliance
elder_data=df_clean_EMS.loc[df_clean_EMS['Facility Name'].str.contains('Elder')]
# %%
elder_data.to_csv('Elderly & Physically Disabled Providers Alliance Total given.csv')

# %%
