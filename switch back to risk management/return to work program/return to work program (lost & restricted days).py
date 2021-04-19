#%%
import pandas as pd
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
import seaborn as sns
import datetime 
#from datetime import datetime
import missingno as msno
import smartsheet
import os
import docx
datt=str(datetime.datetime.now())[:10]


PROJECT_ROOT_DIR = "."
CHAPTER_ID = "return to work program"
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID)
os.makedirs(IMAGES_PATH, exist_ok=True)

def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)
# %%
def read_select_data():
    path='C:/Users/Mr.Goldss/Desktop/switch back to risk management/'
    filename='AdhocLostandR202012077d5bba9883c94257b53f18a6e51aa9e0'
    df=pd.read_excel(path+filename+'.xlsx',sheet_name='Data',skiprows=4)
    df=df[df['Claim Number'].notna()]
    df=df[df['Work Days']>0]
    return df
df=read_select_data()

# %%

lost_time_type_lostdays=df[df['Lost Time Type']=='Lost Days']
lost_df=lost_time_type_lostdays
lost_time_type_restricted=df[df['Lost Time Type']=='Restricted Days']
res_df=lost_time_type_restricted

#%%

#%%
# people LWD only, some workers have more than 1 lost days records
lost_only_worker=lost_df[~lost_df['Claim Number'].isin(res_df['Claim Number'])]
lost_only_worker=pd.DataFrame(lost_only_worker.groupby('Claim Number')['Work Days'].sum())
lost_only_worker.describe()

#%%
def normal_dist_plot(x_min = 1,x_max = 99,mean = 15,std = 13.6):

    import matplotlib.pyplot as plt
    import scipy.stats
    import numpy as np



    x = np.linspace(x_min, x_max, 100)

    y = scipy.stats.norm.pdf(x,mean,std)

    plt.plot(x,y, color='black')

    #----------------------------------------------------------------------------------------#
    # fill area 1

    pt1 = mean + std
    plt.plot([pt1 ,pt1 ],[0.0,scipy.stats.norm.pdf(pt1 ,mean, std)], color='black')

    pt2 = mean - std
    plt.plot([pt2 ,pt2 ],[0.0,scipy.stats.norm.pdf(pt2 ,mean, std)], color='black')

    ptx = np.linspace(pt1, pt2, 10)
    pty = scipy.stats.norm.pdf(ptx,mean,std)

    plt.fill_between(ptx, pty, color='#0b559f', alpha=1.0)

    #----------------------------------------------------------------------------------------#
    # fill area 2

    pt1 = mean + std
    plt.plot([pt1 ,pt1 ],[0.0,scipy.stats.norm.pdf(pt1 ,mean, std)], color='black')

    pt2 = mean + 2.0 * std
    plt.plot([pt2 ,pt2 ],[0.0,scipy.stats.norm.pdf(pt2 ,mean, std)], color='black')

    ptx = np.linspace(pt1, pt2, 10)
    pty = scipy.stats.norm.pdf(ptx,mean,std)

    plt.fill_between(ptx, pty, color='#2b7bba', alpha=1.0)

    #----------------------------------------------------------------------------------------#
    # fill area 3

    pt1 = mean - std
    plt.plot([pt1 ,pt1 ],[0.0,scipy.stats.norm.pdf(pt1 ,mean, std)], color='black')

    pt2 = mean - 2.0 * std
    plt.plot([pt2 ,pt2 ],[0.0,scipy.stats.norm.pdf(pt2 ,mean, std)], color='black')

    ptx = np.linspace(pt1, pt2, 10)
    pty = scipy.stats.norm.pdf(ptx,mean,std)

    plt.fill_between(ptx, pty, color='#2b7bba', alpha=1.0)

    #----------------------------------------------------------------------------------------#
    # fill area 4

    pt1 = mean + 2.0 * std
    plt.plot([pt1 ,pt1 ],[0.0,scipy.stats.norm.pdf(pt1 ,mean, std)], color='black')

    pt2 = mean + 3.0 * std
    plt.plot([pt2 ,pt2 ],[0.0,scipy.stats.norm.pdf(pt2 ,mean, std)], color='black')

    ptx = np.linspace(pt1, pt2, 10)
    pty = scipy.stats.norm.pdf(ptx,mean,std)

    plt.fill_between(ptx, pty, color='#539ecd', alpha=1.0)

    #----------------------------------------------------------------------------------------#
    # fill area 5

    pt1 = mean - 2.0 * std
    plt.plot([pt1 ,pt1 ],[0.0,scipy.stats.norm.pdf(pt1 ,mean, std)], color='black')

    pt2 = mean - 3.0 * std
    plt.plot([pt2 ,pt2 ],[0.0,scipy.stats.norm.pdf(pt2 ,mean, std)], color='black')

    ptx = np.linspace(pt1, pt2, 10)
    pty = scipy.stats.norm.pdf(ptx,mean,std)

    plt.fill_between(ptx, pty, color='#539ecd', alpha=1.0)

    #----------------------------------------------------------------------------------------#
    # fill area 6

    pt1 = mean + 3.0 * std
    plt.plot([pt1 ,pt1 ],[0.0,scipy.stats.norm.pdf(pt1 ,mean, std)], color='black')

    pt2 = mean + 10.0 *std
    plt.plot([pt2 ,pt2 ],[0.0,scipy.stats.norm.pdf(pt2 ,mean, std)], color='black')

    ptx = np.linspace(pt1, pt2, 10)
    pty = scipy.stats.norm.pdf(ptx,mean,std)

    plt.fill_between(ptx, pty, color='#89bedc', alpha=1.0)

    #----------------------------------------------------------------------------------------#
    # fill area 7

    pt1 = mean - 3.0 * std
    plt.plot([pt1 ,pt1 ],[0.0,scipy.stats.norm.pdf(pt1 ,mean, std)], color='black')

    pt2 = mean - 10.0 * std
    plt.plot([pt2 ,pt2 ],[0.0,scipy.stats.norm.pdf(pt2 ,mean, std)], color='black')

    ptx = np.linspace(pt1, pt2, 10)
    pty = scipy.stats.norm.pdf(ptx,mean,std)

    plt.fill_between(ptx, pty, color='#89bedc', alpha=1.0)

    #----------------------------------------------------------------------------------------#

    plt.grid()

    plt.xlim(x_min,x_max)
    plt.ylim(0,0.25)

    plt.title('Loss days normal distribution',fontsize=10)

    plt.xlabel('Days')
    plt.ylabel('Percentage')

    #plt.savefig("normal_distribution_2.png")
    plt.show()
#%%
normal_dist_plot(x_min = 1,x_max = 99,mean = int(lost_only_worker.describe().loc['mean']),std = int(lost_only_worker.describe().loc['std']))
save_fig('curr')

#%%
normal_dist_plot(x_min = 1,x_max = 99,mean = 51,std =20.3)
#%%
#自己尝试不同work days 应该占有的百分比
#<=19
aa=lost_only_worker[(lost_only_worker['Work Days']<=19)]
print(aa.describe())
normal_dist_plot(x_min = 1,x_max = 30,mean = 10.78,std = 4.32)


bb=lost_only_worker[(lost_only_worker['Work Days']>19)]
print(bb.describe())
normal_dist_plot(x_min = 11,x_max = 99,mean = 38.84,std = 21.15)




#%%
# people RWD only, some workers have more than 1 restricted days records
res_only_worker=res_df[~res_df['Claim Number'].isin(lost_df['Claim Number'])]

duplicate_res_claimNumber=pd.DataFrame(res_only_worker['Claim Number'].value_counts()==2).reset_index()
duplicate_res_number=duplicate_res_claimNumber[duplicate_res_claimNumber['Claim Number']==True]['index']

res_only_worker=res_only_worker.groupby('Claim Number')['Work Days'].sum()

#%%
# people both LWD & RWD
res_and_lost_worker=df[(df['Claim Number'].isin(lost_df['Claim Number'])) & (df['Claim Number'].isin(res_df['Claim Number']))]

print(res_and_lost_worker.shape)


#%%

#continue analysis people noth LWD & RWD
#for people with both LWD and RWD, we need take care of Is Accommodated  and Transitional Duty missing value
#situation 1: if the Lost Time Type is Restricted Days and Is Accommodated is missing, fill with True
#situation 2: if the Lost Time Type is Lost Days and Is Accommodated is missing, fill with False
#This is a temporary fillna solution, for most accurate result, I need to assign this back to Tim then have
#somebody to fill those value in Origami Risk.
def fill_missing_for_both_people():
        
    res_and_lost_worker_L1=res_and_lost_worker.loc[res_and_lost_worker['Lost Time Type']=='Lost Days']
    res_and_lost_worker_L1['Is Accommodated']=res_and_lost_worker_L1['Is Accommodated'].fillna(False)
    res_and_lost_worker_L1['Transitional Duty']=res_and_lost_worker_L1['Transitional Duty'].fillna(False)


    #Question: for here I just input with True which is not accurate, could be False, really depend on 
    #situation, but ignore it for now and talk with Tim for a final decision

    res_and_lost_worker_R2=res_and_lost_worker.loc[res_and_lost_worker['Lost Time Type']=='Restricted Days']
    res_and_lost_worker_R2['Is Accommodated']=res_and_lost_worker_R2['Is Accommodated'].fillna(True)
    #res_and_lost_worker[(res_and_lost_worker['Lost Time Type']=='Restricted Days') & (res_and_lost_worker['Is Accommodated'].isna())]

    fill_res_and_lost_worker=pd.concat([res_and_lost_worker_L1,res_and_lost_worker_R2])
    return fill_res_and_lost_worker


res_and_lost_worker=fill_missing_for_both_people()


#%%
#Employees who are released to restricted duty where the employer is unable 
# to accommodate their restrictions then continues to count as lost work days. 
#   the following condition used to filter people who considered as Restricted days but actually is lost work days
cond1=res_and_lost_worker['Lost Time Type']=='Restricted Days'
cond2=res_and_lost_worker['Is Accommodated']==False
cond3=res_and_lost_worker['Transitional Duty']==False

wrong_res_people=res_and_lost_worker[cond1 & cond2 & cond3]

# those people considered as lost days only
# so that we know there are 14 records will be consider as Lost Days
# There are 11 unique "Claim Number"
wrong_res_people.groupby('Claim Number')[['Indemnity Paid','Work Days']].agg({'Indemnity Paid':'first','Work Days':'sum'})
print(wrong_res_people.groupby('Claim Number')[['Indemnity Paid','Work Days']].agg({'Indemnity Paid':'first','Work Days':'sum'}))
#%%
#L_only contain only Lost Days only claim numbers
L_only=list(lost_only_worker['Claim Number'].unique())
L_only.extend(wrong_res_people['Claim Number'].unique())
print(len(L_only))
R_only=list(res_only_worker['Claim Number'].unique())
print(len(R_only))

print(len(L_only)+len(R_only))
print(df['Claim Number'].nunique())

# we only have 21 person in both L&R

#%%
#res_and_lost_worker has 72 record here so that we have 58 records that shall be considered both L and R
# in total we have 32 unique both L&R case and we already know 14 of them will be consider as lost days 
# so that we only have
print(res_and_lost_worker.groupby('Claim Number')[['Indemnity Paid','Work Days']].agg({'Indemnity Paid':'first','Work Days':'sum'}).shape)






# %%
print('lost day only records')
print(lost_only_worker.shape)

print('restricted day only records')
print(res_only_worker.shape)

print('Both lost dat and restricted dat records')
print(res_and_lost_worker.shape)

print('compare with the original data')
print(df.shape)
# %%
#lost only people total lost days
print(lost_only_worker.describe())
print()
print(res_only_worker.describe())
#print(res_and_lost_worker.shape)
# %%


#question: how to deal with data that 'Is Accommodated' is NA
res_only_worker[res_only_worker['Is Accommodated'].isna()]
#question2: how to deal with data that 'Transitional Duty' is NA
res_only_worker[res_only_worker['Transitional Duty'].isna()]

#%%
