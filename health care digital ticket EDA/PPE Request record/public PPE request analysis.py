#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt 
import os
from datetime import date

today = date.today()

# %%
PROJECT_ROOT_DIR = "."
CHAPTER_ID = "ppe request chart"
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID)
os.makedirs(IMAGES_PATH, exist_ok=True)

def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)
#%%
path='C:/Users/Mr.Goldss/Desktop/health care digital ticket EDA/PPE Request record/'
filename='PPE MASTER'
ppe_request=pd.read_excel(path+filename+'.xlsx',sheet_name='PPE Form Response')

ppe_request.shape

needed_col=['Timestamp'	,'Facility Name','Organization Type','City',	'Zip Code',
'How many staff do you have working in your facility on a regular basis?',
'N95 Respirators - Request',	'Face masks or surgical masks - Request',
'Face shields - Request',	'Gloves (various sizes)  - Request',	'Gloves ', 	
'Isolation gowns - Request','N95 Respirators - Burn Rate',	
'Face masks or surgical masks - Burn Rate',	'Face shields - Burn Rate',	
'Gloves - Burn Rate ', 	'Isolation gowns - Burn Rate']
#%%
ppe_request=ppe_request[needed_col]

ppe_request.columns=list(map(str.lower,['Timestamp'	,'Facility Name','Organization Type','City','Zip Code',
'Number of workers',
'N95 Request',	'Face masks or surgical masks Request',
'Face shields Request',	'Gloves Request',	'Gloves', 	
'gowns Request','N95 Burn Rate',	
'Face masks or surgical masks Burn Rate',	'Face shields Burn Rate',	
'Gloves Burn Rate ', 	'gowns Burn Rate']))
ppe_request.head()

# %%
ppe_request.dropna(subset=['timestamp'],inplace=True)
ppe_request.shape
ppe_request['gloves request'].fillna(0,inplace=True)
ppe_request['gloves'].fillna(0,inplace=True)
ppe_request['gloves']=ppe_request['gloves'].replace('S,m,l,xl,xxl',0)
ppe_request['gloves request']=ppe_request['gloves request'].astype(int)
ppe_request['gloves request']=ppe_request['gloves'].astype(int)
ppe_request['gloves request combined']=ppe_request['gloves request']+ppe_request['gloves']
ppe_request.drop(['gloves request', 'gloves'], axis=1, inplace=True)
#change the column order
#%%
ppe_request=ppe_request[['timestamp', 'facility name', 'organization type', 'city', 'zip code',
       'number of workers', 'n95 request',
       'face masks or surgical masks request', 'face shields request',
       'gowns request', 'gloves request combined','n95 burn rate',
       'face masks or surgical masks burn rate', 'face shields burn rate',
       'gloves burn rate ', 'gowns burn rate']]
# %%
'''date related'''

ppe_request['month']=ppe_request['timestamp'].dt.month
ppe_request['week_of_year']=ppe_request['timestamp'].dt.week
ppe_request.head()
#%%
ppe_request['n95 request']=ppe_request['n95 request'].replace('Have none of these',0)
ppe_request['n95 request'].fillna(0,inplace=True)
ppe_request['n95 request']=ppe_request['n95 request'].astype(int)
cat_request=['n95 request',
       'face masks or surgical masks request', 'face shields request',
       'gowns request', 'gloves request combined']
ppe_request['gowns request']=ppe_request['gowns request'].replace('784/30',0)
ppe_request['gowns request']=ppe_request['gowns request'].replace('00',0)
for i in cat_request:
       ppe_request[i].fillna(0,inplace=True)

       ppe_request[i]=ppe_request[i].astype(int)



ppe_request['request total']=ppe_request['n95 request']+ppe_request['face masks or surgical masks request']+ppe_request['face shields request']+\
       ppe_request['gowns request']+ppe_request['gloves request combined']


#%%

cat_request=['n95 request',
       'face masks or surgical masks request', 'face shields request',
       'gowns request', 'gloves request combined','request total']
for i in cat_request:
       fig, ax1, = plt.subplots()
       fig.set_size_inches(25, 12)
       sns.lineplot(x='month',y=i,\
       data=ppe_request,ax=ax1,estimator ='mean')
       plt.xticks(fontsize=18,rotation=90)
       plt.xlabel('Month', fontsize=30)
       plt.ylabel('Amount', fontsize=30)
       plt.title('mean '+i+' by month', fontsize=30)
       
       plt.tight_layout() 
       save_fig(i+str(today))
       plt.show()






# %%
'''log transfer'''
pic=ppe_request.copy()

for i in cat_request:
       pic[i]=np.log(pic[i])

       fig, ax, = plt.subplots()
       fig.set_size_inches(25, 12)
       ax=sns.boxplot(x='month',y=i,data=pic)
       ax=plt.xticks(rotation=90,fontsize=20)
       ax=plt.yticks(fontsize=20)
       ax=plt.xlabel('Month')
       ax=plt.title(i,fontsize=30)
       ax=plt.ylabel('Log Total Amount')
       plt.tight_layout() 
       plt.show()
#%%

fig, ax, = plt.subplots()
fig.set_size_inches(25, 12)
ax=sns.lineplot(y='n95 request',
       x='month',data=ppe_request,\
              ax=ax1,color='#E55C94')
ax=plt.xticks(rotation=90,fontsize=20)
ax=plt.yticks(fontsize=20)
ax=plt.xlabel('Month')
ax=plt.title(i,fontsize=30)
ax=plt.ylabel('Total Amount')
plt.tight_layout() 
plt.show()

# %%


# %%
