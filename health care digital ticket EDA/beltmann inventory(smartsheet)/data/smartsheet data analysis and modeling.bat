
import datetime
import pandas as pd
import gspread
import matplotlib
import seaborn as sns
import matplotlib.pyplot as plt
import string

path='C:/Users/Mr.Goldss/Desktop/beltmann inventory(smartsheet)/data/'
filename='MARICOPA COUNTY COMPLETE INVENTORY TRANSACTIONS'
need_col=['PROJECT','CATEGORY','ITEM/ SKU','DESCRIPTION','SOURCE','SHIPPER'
,'CASE COUNT','# INNER CASE CTN',	'# PER INNER CTN','TOTAL QTY',
'SERVICE DATE',	'BALANCE',	'CUSTOMER NOTES','qty out','full pallet out',	'RELEASE DATE','Workorder Nbr']
df_smart=pd.read_excel(path+filename+'.xlsx')
df_smart=df_smart[need_col]



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
       df2=df2[df2['customer notes']!='Vault 12 week']
       df_smt=pd.concat([df1, df2], axis=1)
       return df_smt



df_smt=smart_preprocessing(df_smart)






cond_still_in_storage=df_smt['full pallet out']==0

cond2_check_andy=df_smt['service date']<='9/8/2020'


smart_stock=df_smt[cond_still_in_storage & cond2_check_andy].groupby('category')['balance'].sum().sort_values(ascending=False)


smart_stock.reset_index()

smart_stock=pd.DataFrame(smart_stock)




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



#email_list=['andrew.weflen@maricopa.gov','zjin24@asu.edu','richard.langevin@maricopa.gov','gerry.gill@maricopa.gov','Matthew.Melendez@maricopa.gov','Ashley.Miranda@maricopa.gov']

send_email_attach(
    from_who='zixiangjin921@gmail.com',
    to_who=i,subject_name='Beltmann Dailt inventory update',
    content='Inventory data exclude Vault 12 week. data from smartsheet, update date 7/16/2020 9:21 am',
    attach_file_name='beltmann inventory.png',
    file_path='C:/Users/Mr.Goldss/Desktop/beltmann inventory(smartsheet)/image/beltmann inventory 20200716 091907832611.png',
    password='Intelcorei7')