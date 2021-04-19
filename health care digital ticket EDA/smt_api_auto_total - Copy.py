#%%
def finish_them_all():

        

    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    import datetime 
    #from datetime import datetime
    import missingno as msno
    import smartsheet
    import os
    datt=str(datetime.datetime.now())


    import gspread
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

    path='C:/Users/Mr.Goldss/Desktop/health care digital ticket EDA/'
    filename='10 1 2020 Facility Itemized Tracker (1)'
    df=pd.read_excel(path+filename+'.xlsx',sheet_name='New ALL')


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

    df_clean_EMS['ticket_unique_id']=list(map(str,df_clean_EMS['Delivered Date']))+df_clean_EMS['Facility Name']

    df_dir_concat=df_clean_EMS[['Type','Total Amount','Delivered Date']]
    df_dir_concat.columns=['category','qty out','release date']


    '''
    Beltmann data api conn

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

    df_smt['release date']=df_smt['release date'].replace('M-07162020-8','2020-07-16')
    
    df_smt['release date']=df_smt['release date'].replace('M-10232020-4','2020-10-23')
    df_smt['release date']=df_smt['release date'].replace('FACE MASK Taiwan cloth','2020-08-27')
    df_smt['release date']=df_smt['release date'].replace('Hand sanitizer 33 oz per bottle','2020-08-27')
    df_smt['release date']=pd.to_datetime(df_smt['release date'])


    #cond_still_in_storage=df_smt['full pallet out']==0

    df_smt.loc[df_smt['balance']=='#UNPARSEABLE','balance']=0
    df_smt['category']=df_smt['category'].str.lower()

    df_smt=df_smt[df_smt['release to cust:'].notna()]
    df_smt['release date'].fillna(method='ffill',inplace=True)


    '''Daily Total Give out'''

    today_date=datt[:10]
    df_today=df_smt[df_smt['release date']==today_date]
    df_today=df_today.groupby('category')['qty out'].sum().sort_values(ascending=False)
    df_today.to_csv(datt[:10]+' Total give out.csv')
    for i in email_list2:
        send_email_attach(
        from_who='zixiangjin921@gmail.com',
        to_who=i,
        subject_name=datt[:10]+' Total giveout.csv',
        content='update time: '+datt,
        attach_file_name=datt[:10]+' PPE total giveout.csv',
        file_path='C:/Users/Mr.Goldss/Desktop/health care digital ticket EDA/'+datt[:10]+' Total give out.csv',
        password='Intelcorei7')















    #replace smartsheet category with dirks standard or the other way around
    cat_out=df_smt[df_smt['qty out']>0][['category','qty out','release date']]
    beltmann_cat=cat_out.category.unique()


    print(beltmann_cat)
    replace_cat={'medical or surgical masks':'surgical mask','alternative gowns':'alt. gowns','kn95 masks':'kn95',
    'cleaning/disinfectant':'sanitizer','eye protection':'goggles','face shield':'face shield','gloves':'gloves',
    'n95 masks':'n95','hand sanitizer/hygiene':'sanitizer',}

    cat_out.category=cat_out.category.replace(replace_cat)



    df_both=pd.concat([cat_out,df_dir_concat])

    df_both['release date']=pd.to_datetime(df_both['release date'])

    df_both['week_of_year']=df_both['release date'].dt.strftime('%U')

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





    '''total given both inventory data '''
    df_total_out_both_alltime=df_both.groupby('category')['qty out'].sum().sort_values(ascending=False)
    df_total_out_both_alltime.to_csv('Covid-19 PPE total giveout.csv')


    '''EMAIL'''


    for i in email_list2:
        send_email_attach(
        from_who='zixiangjin921@gmail.com',
        to_who=i,
        subject_name='Total giveout update',
        content='update time: '+datt,
        attach_file_name='Covid-19 PPE total giveout.csv',
        file_path='C:/Users/Mr.Goldss/Desktop/health care digital ticket EDA/Covid-19 PPE total giveout.csv',
        password='Intelcorei7')

finish_them_all()
# %%
#finish_them_all()
import schedule
import time
schedule.every().day.at("16:00").do(finish_them_all)
while True:
    schedule.run_pending()
    time.sleep(1)
# %%

# %%
import time
time.gmtime(0)
# %%
import folium
map=folium.Map(location=[32.4,-111])
# %%
finish_them_all()
# %%
