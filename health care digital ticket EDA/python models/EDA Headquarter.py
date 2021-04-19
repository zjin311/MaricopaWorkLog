#%%
'''Exploratory data analysis can never be the
whole story, but nothing else can serve as the founadtion
stone. I wholeheartedly agree with this'''


#001 ctrl+/  comment out all selected rows
#002 select rows then <tab>  tab space all those rows----> shift+tab to bounce back




import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt 
import missingno as msno
#%%
path='C:/Users/Mr.Goldss/Desktop/health care digital ticket EDA/'
filename='7_23_2020 Facility Itemized Tracker (1)'
df=pd.read_excel(path+filename+'.xlsx',sheet_name='New ALL')
#%%
df.head()
df.describe()
df.corr()
# %%
#visualize missing partern
msno.matrix(df)
msno.bar(df)
msno.heatmap(df)

#%%
#Note:  std**2==var
np.var()
np.std()
np.sqrt()

#Covariance indicates the level to 
# which two variables vary together
#take-->[0,1] as the correlation
np.cov(x,y)
np.corrcoef(x,y) #----->Pearson correlation is easier to be interpret
#but i prefer this version:
df_clean_EMS[['QTY Per UOM','Total Amount']].corr()
#corr heat map
correlation = df.corr()
plt.figure(figsize=(10,10))
sns.heatmap(correlation, vmax=1, square=True,annot=True,cmap='viridis')

plt.title('Correlation between different fearures')


#%%
np.cov(df_clean_EMS['QTY Per UOM'],df_clean_EMS['Total Amount'])
np.cov([1,2,3,4,5],[5,4,3,2,1])




#%%
#date related
# Extracting Date attributes
df['Date'] = pd.to_datetime(df['Date'])
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day
df['Day_of_Week'] = df['Date'].dt.dayofweek
df.drop('Date', axis = 1, inplace = True)

#%%

app_per_cat = dict(df['Category'].value_counts())
sns.barplot(list(app_per_cat.values()), list(app_per_cat.keys()),orient= 'h')
plt.title("Number of apps per category")
plt.show()













#%%
#hist is useful when analysis numerical column distribution
#bins=[0,10,20,30,40,50,60,70,80,90]---> bin edges
#use seaboen default style by sns.set()

#The "square root rule" is a commonly-used rule of thumb 
# for choosing number of bins: choose the number of bins to be 
# the square root of the number of samples. 
bin_edges=[0,500,1000,1500,2000,2500,3000]  
#----> use this to define bin edge
n_bins=int(np.sqrt(df.shape[0]))
#----> if you dont know how many bins to present use the: square root rule

sns.set()
ax=plt.hist(df['col'],bins=20,density=,cumulative=True)
ax=plt.xlabel('name of the x axis')
ax=plt.ylabel('name of the y axis')
plt.tight_layout() 
plt.show()





#%%
fig, ax, = plt.subplots()
#In daily work some categorical column could have a lot of category  .isin([]) is a
#good way to corp the data in to a smaller piece
dt=df_clean_EMS[df_clean_EMS['Type'].isin(['n95','surgical mask'])]
ax=sns.swarmplot(x='Type',y='Total Amount',data=dt)
fig.set_size_inches(25, 12)
ax=plt.xlabel('name of the x axis')
ax=plt.ylabel('name of the y axis')
plt.tight_layout() 
plt.show()

'''swarmplot problem： data points overlapping: is there a better way?'''

#%%
#YES-->ECDF(empirical cumulative distribution function)
#Even though I am not familiar with ECDF, but it is really helpful
#you should try to apply ECDF to every dataset!
def ecdf(data):
    """Compute ECDF for a one-dimensional array of measurements."""
    # Number of data points: n
    n = len(data)

    # x-data for the ECDF: x
    x = np.sort(data)

    # y-data for the ECDF: y
    y = np.arange(1, n+1) / n

    return x, y
    
# Compute ECDF for versicolor data: x_vers, y_vers
x_vers, y_vers = ecdf(df['col'])
# Generate plot
plt.plot(x_vers,y_vers,marker = '.' , linestyle = 'none')
plt.legend(('setosa', 'versicolor', 'virginica'), loc='lower right') 
# now you know how to edit legend babe:)---> legend name same as plot number
plt.xlabel('label of x axis')
plt.ylabel('ECDF')
plt.tight_layout() 
plt.show()

#||
#||    this is a good example! Look at me :)
#VV
def ecdf_example_ppe():
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
#pdf=> probability density function

#%%
'''boxplot'''
fig, ax, = plt.subplots()
fig.set_size_inches(25, 12)
data_cond_amount=df_clean_EMS[df_clean_EMS['Total Amount']<=5000]
ax=sns.boxplot(x='Type',y='Total Amount',data=data_cond_amount)
ax=plt.xticks(rotation=90,fontsize=12)
ax=plt.xlabel('Type')
ax=plt.ylabel('Total Amount')
plt.tight_layout() 
plt.show()

#%%
fig, ax1, = plt.subplots()

sns.scatterplot(y='Total Amount',
            x='week_of_year',data=df_clean_EMS[df_clean_EMS['Total Amount']<=5000],\
                ax=ax1,color='#E55C94')
plt.tight_layout() 
plt.show()

#%%
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

#%%
'''study pandas profiling'''
#https://pandas-profiling.github.io/pandas-profiling/docs/master/rtd/pages/advanced_usage.html#general-settings
import numpy as np
import pandas as pd
from pandas_profiling import ProfileReport

df = pd.DataFrame(
    np.random.rand(100, 5),
    columns=['a', 'b', 'c', 'd', 'e']
)

profile = ProfileReport(df, title='Pandas Profiling Report')

# %%
profile = ProfileReport(df, title='Pandas Profiling Report', explorative=True)

# %%
#https://www.kaggle.com/vbmokin/automatic-eda-with-pandas-profiling-2-9-07-2020
profile.to_widgets()
profile.to_file("your_report.html")


# %%
