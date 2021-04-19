#%%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# %%
dirr='C:/Users/Mr.Goldss/Desktop/health care digital ticket EDA/'
df=pd.read_excel(dirr+'1-19-2021 gerry count Daily Report.xlsx',None)

# %%
df_full=df.copy()
df_full = pd.DataFrame()
for _, sheet in df.items(): df_full = df_full.append(sheet) 
df_full=df_full.reset_index()


df_full=df_full[['Date', 'Date ','City ', 'Facility ']]

df_full.columns=['Date1', 'Date2','City', 'Facility']

df_full=df_full.fillna('Missing')
date1_df=df_full[df_full['Date1']!='Missing']
date2_df=df_full[df_full['Date2']!='Missing']
# %%
def combine_date(df_full):
    date1_df=df_full[df_full['Date1']!='Missing'][['Date1','City','Facility']]
    date2_df=df_full[df_full['Date2']!='Missing'][['Date2','City','Facility']]
    date2_df.columns=['Date1','City','Facility']
    dddd=pd.concat([date1_df,date2_df])
    return dddd
df_full=combine_date(df_full)


df_full.Date1=pd.to_datetime(df_full.Date1)
df_full=df_full[df_full['City']!='Missing']
# %%

df_full.groupby('Date1')['Facility'].count()
# %%
df_draw=pd.DataFrame(df_full.groupby('Date1')['Facility'].count())
df_draw=df_draw.reset_index()
df_draw['year']=df_draw['Date1'].dt.year
df_draw['month']=df_draw['Date1'].dt.month
df_draw['week']=df_draw['Date1'].dt.week
#%%
df_draw['week']=np.where(df_draw['year']==2021,df_draw['week']+52,df_draw['week'])
#%%
#df_draw.to_csv('gerry count chart.csv')



#%%
df_draw=pd.DataFrame(df_draw.groupby('week')['Facility'].sum()).reset_index()

#%%
fig, ax = plt.subplots(figsize=(15, 6))
palette = sns.color_palette("ch:2.5,-.2,dark=.3", 10)
sns.lmplot(x='week', y="Facility", data=df_draw,)

#%%
fig, ax = plt.subplots(figsize=(15, 9))
palette = sns.color_palette("ch:2.5,-.2,dark=.3", 10)
sns.regplot(x='week', y="Facility", data=df_draw,ax=ax)
ax.set_title('Total Request by week', fontsize = 20, loc='center', fontdict=dict(weight='bold'))
ax.set_xlabel('Number of week', fontsize = 16, fontdict=dict(weight='bold'))
ax.set_ylabel('Applied Facility', fontsize = 16, fontdict=dict(weight='bold'))
plt.show()
# %%
df = pd.DataFrame({"A":[1,2,3],
                   "B":[4,5,7]})
fig, ax = plt.subplots(1,1)
#sns.lmplot("A","B",df,ax=ax)
sns.regplot("A","B",df,ax=ax) # regplot works as intended
plt.show()
# %%
df_full.to_csv('facility request count.csv')
# %%
