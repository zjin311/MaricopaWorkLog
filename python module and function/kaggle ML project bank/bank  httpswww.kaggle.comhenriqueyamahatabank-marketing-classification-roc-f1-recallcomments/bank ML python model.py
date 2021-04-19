    
#%%

#

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
import warnings
warnings.filterwarnings('ignore')
#%%

path='C:/Users/Mr.Goldss/Desktop/python module and function/kaggle ML project/bank problem'
bank = pd.read_csv(path+'/bank-additional-full.csv', sep = ';')
y=pd.get_dummies(bank['y'],columns=['y'],prefix='y',drop_first=True)
#no label 0 ;  yes label 1



# %%
'''preveiw dataset'''
bank.info()
bank.columns
#%%

'''our dataset have so many columns: we need to seperate it into a few parts by column'''

#part1 bank client
bank_client=bank.iloc[:,:7]
bank_client['y']=bank['y']
bank_client.head()


# %%
'''analysis by order:==>     categorical & numerical '''
# knowing the categorical variables
print('Jobs:\n', bank_client['job'].value_counts()/bank_client.shape[0])
print()
print('Marital:\n', bank_client['marital'].value_counts()/bank_client.shape[0])
print()
print('Education:\n', bank_client['education'].value_counts()/bank_client.shape[0])
print()
print('Default:\n', bank_client['default'].value_counts()/bank_client.shape[0])
print()
print('Housing:\n', bank_client['housing'].value_counts()/bank_client.shape[0])
print()
print('Loan:\n', bank_client['loan'].value_counts()/bank_client.shape[0])
print()
# %%
# knowing the numerical variables:age
'''#Calculate the outliers:
  # Interquartile range, IQR = Q3 - Q1
  # lower 1.5*IQR whisker = Q1 - 1.5 * IQR 
  # Upper 1.5*IQR whisker = Q3 + 1.5 * IQR'''

stat=bank_client['age'].describe()
print('age stat:',stat)
Q1=stat['25%']
Q3=stat['75%']
IQR=stat['75%']-stat['25%']
Upper=Q3+1.5*IQR
Lower=Q1-1.5*IQR
print('upper:'+str(Upper)+'\n'+'lower:'+ str(Lower))

#percentage of upper outlier
out_perc=bank_client[bank_client['age']>Upper].shape[0]/bank_client.shape[0]

print('Outliers are:', round(out_perc*100,2), '%')

# %%

#Age countplot
fig, ax = plt.subplots()
fig.set_size_inches(20, 8)
sns.countplot(x = 'age',  hue='y', data = bank_client)
plt.xlabel('Age', fontsize=15)
plt.ylabel('Count', fontsize=15)
ax.set_title('Age Count Distribution', fontsize=15)
sns.despine()


# %%
#age boxplot and histplot
#boxplot and histplot are 2 ways to show distribution
fig, (ax1, ax2) = plt.subplots(nrows = 1, ncols = 2, figsize = (13, 5))
sns.boxplot(x = 'age', data = bank_client, orient = 'v',ax = ax1)
ax1.set_xlabel('People Age', fontsize=15)
ax1.set_ylabel('Age', fontsize=15)
ax1.set_title('Age Distribution', fontsize=15)
ax1.tick_params(labelsize=15)

sns.distplot(bank_client['age'], ax = ax2)
sns.despine(ax = ax2)
ax2.set_xlabel('Age', fontsize=15)
ax2.set_ylabel('Occurence', fontsize=15)
ax2.set_title('Age x Ocucurence', fontsize=15)
ax2.tick_params(labelsize=15)

plt.subplots_adjust(wspace=0.5)  # 两图间距离
plt.tight_layout()               #也不能让两个图分的太远
# %%
'''calculate coef variation'''
# Calculating some values to evaluete this independent variable
print('MEAN:', round(bank_client['age'].mean(), 1))
# A low standard deviation indicates that the data points tend to be close to the mean or expected value
# A high standard deviation indicates that the data points are scattered
print('STD :', round(bank_client['age'].std(), 1))
# I thing the best way to give a precisly insight abou dispersion is using the CV (coefficient variation) (STD/MEAN)*100
#    cv < 15%, low dispersion
#    cv > 30%, high dispersion
print('CV  :',round(bank_client['age'].std()*100/bank_client['age'].mean(), 1), ', High middle dispersion')

'''Conclusion about AGE, in my opinion due to almost high 
dispersion and just looking at this this graph we cannot conclude
 if age have a high effect to our variable y, need to keep 
 searching for some pattern. high middle dispersion means we have 
 people with all ages and maybe all of them can subscript a term 
 deposit, or not. The outliers was calculated, so my thinking is 
 fit the model with and without them'''
# %%

#job
fig, ax = plt.subplots()
fig.set_size_inches(20, 8)
sns.countplot(x = 'job', hue='y',data = bank_client)
plt.xlabel('job', fontsize=15)
plt.ylabel('Count', fontsize=15)
ax.set_title('job Count Distribution', fontsize=15)
sns.despine()

# %%
#marital
fig, ax = plt.subplots()
fig.set_size_inches(20, 8)
sns.countplot(x = 'marital', hue='y',data = bank_client)
plt.xlabel('marital', fontsize=15)
plt.ylabel('Count', fontsize=15)
ax.set_title('marital Count Distribution', fontsize=15)
sns.despine()

# %%
#education
fig, ax = plt.subplots()
fig.set_size_inches(20, 8)
sns.countplot(x = 'education', hue='y',data = bank_client)
plt.xlabel('educationge', fontsize=15)
plt.ylabel('Count', fontsize=15)
ax.set_title('education Count Distribution', fontsize=15)
sns.despine()

# %% CHART FOR
#'default', 'housing', 'loan'

fig, (ax1, ax2,ax3) = plt.subplots(nrows = 1, ncols = 3, figsize = (13, 5))
sns.countplot(x = 'default', data = bank_client, hue='y',orient = 'v', ax = ax1,order = ['no', 'unknown', 'yes'])
ax1.set_xlabel('default ', fontsize=15)
ax1.set_ylabel('count', fontsize=15)
ax1.set_title('default Distribution', fontsize=15)
ax1.tick_params(labelsize=15)

sns.countplot(x = 'housing', data = bank_client, hue='y',orient = 'v', ax = ax2,order = ['no', 'unknown', 'yes'])
ax1.set_xlabel('housing', fontsize=15)
ax1.set_ylabel('count', fontsize=15)
ax1.set_title('housing Distribution', fontsize=15)
ax1.tick_params(labelsize=15)

sns.countplot(x = 'loan', data = bank_client, hue='y',orient = 'v', ax = ax3,order = ['no', 'unknown', 'yes'])
ax1.set_xlabel('loan', fontsize=15)
ax1.set_ylabel('count', fontsize=15)
ax1.set_title('loan Distribution', fontsize=15)
ax1.tick_params(labelsize=15)

plt.subplots_adjust(wspace=0.5)  # 两图间距离
plt.tight_layout()               #也不能让两个图分的太远

# %% 
#NUMBERS FOR:'default', 'housing', 'loan'
lst=['default', 'housing', 'loan']
for i in lst:

    print(i,'\n',bank_client[i].value_counts())
    print()


# %%
#Im done with bank client data exploration not its time to encode them
from sklearn.preprocessing import LabelEncoder
labelencoder_X = LabelEncoder()
bank_client['job']      = labelencoder_X.fit_transform(bank_client['job']) 
bank_client['marital']  = labelencoder_X.fit_transform(bank_client['marital']) 
bank_client['education']= labelencoder_X.fit_transform(bank_client['education']) 
bank_client['default']  = labelencoder_X.fit_transform(bank_client['default']) 
bank_client['housing']  = labelencoder_X.fit_transform(bank_client['housing']) 
bank_client['loan']     = labelencoder_X.fit_transform(bank_client['loan']) 


'''
$$$ Manualy way to convert Categorical in Continuous $$$
 
bank_client['job'].replace(['housemaid' , 'services' , 'admin.' , 'blue-collar' , 'technician', 'retired' , 'management', 'unemployed', 'self-employed', 'unknown' , 'entrepreneur', 'student'] , [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], inplace=True)

bank_client['education'].replace(['basic.4y' , 'high.school', 'basic.6y', 'basic.9y', 'professional.course', 'unknown' , 'university.degree' , 'illiterate'], [1, 2, 3, 4, 5, 6, 7, 8], inplace=True)

bank_client['marital'].replace(['married', 'single', 'divorced', 'unknown'], [1, 2, 3, 4], inplace=True)

bank_client['default'].replace(['yes', 'no', 'unknown'],[1, 2, 3], inplace=True)

bank_client['housing'].replace(['yes', 'no', 'unknown'],[1, 2, 3], inplace=True)

bank_client['loan'].replace(['yes', 'no', 'unknown'],[1, 2, 3], inplace=True)

----------------------------------------------------------------------------------------

$$$ A way to Converting Categorical variables using dummies if you judge necessary $$$

--prefix used to give a name after the column is transfered.

bank_client = pd.get_dummies(data = bank_client, columns = ['job'] , prefix = ['job'] , drop_first = True)

bank_client = pd.get_dummies(data = bank_client, columns = ['marital'] , prefix = ['marital'] , drop_first = True)

bank_client = pd.get_dummies(data = bank_client, columns = ['education'], prefix = ['education'], drop_first = True)

bank_client = pd.get_dummies(data = bank_client, columns = ['default'] , prefix = ['default'] , drop_first = True)

bank_client = pd.get_dummies(data = bank_client, columns = ['housing'] , prefix = ['housing'] , drop_first = True)

bank_client = pd.get_dummies(data = bank_client, columns = ['loan'] , prefix = ['loan'] , drop_first = True)
'''

# %%
#function to creat group of ages, this helps because we have 78 differente values here
def age(dataframe):
    dataframe.loc[dataframe['age'] <= 32, 'age'] = 1
    dataframe.loc[(dataframe['age'] > 32) & (dataframe['age'] <= 47), 'age'] = 2
    dataframe.loc[(dataframe['age'] > 47) & (dataframe['age'] <= 70), 'age'] = 3
    dataframe.loc[(dataframe['age'] > 70) & (dataframe['age'] <= 98), 'age'] = 4
           
    return dataframe

age(bank_client)

# %%
fig, ax = plt.subplots()
fig.set_size_inches(20, 8)
sns.countplot(x = 'age',  hue='y', data = bank_client)
plt.xlabel('Age', fontsize=15)
plt.ylabel('Count', fontsize=15)
ax.set_title('Age Count Distribution', fontsize=15)
plt.xticks(rotation=45)
sns.despine()

# %%
# Slicing DataFrame to treat separately, make things more easy
bank_related = bank.iloc[: , 7:11]
bank_related.head()

# %%
for i in bank_related.columns:
    print(i,'\n',bank_related[i].value_counts())
    print()

# %%
fig, (ax1, ax2) = plt.subplots(nrows = 1, ncols = 2, figsize = (13, 5))
sns.boxplot(x = 'duration', data = bank_related, orient = 'v', ax = ax1)
ax1.set_xlabel('Calls', fontsize=10)
ax1.set_ylabel('Duration', fontsize=10)
ax1.set_title('Calls Distribution', fontsize=10)
ax1.tick_params(labelsize=10)

sns.distplot(bank_related['duration'], ax = ax2)
sns.despine(ax = ax2)
ax2.set_xlabel('Duration Calls', fontsize=10)
ax2.set_ylabel('Occurence', fontsize=10)
ax2.set_title('Duration x Ocucurence', fontsize=10)
ax2.tick_params(labelsize=10)

plt.subplots_adjust(wspace=0.5)
plt.tight_layout() 

# %%
'''
https://www.kaggle.com/henriqueyamahata/bank-marketing-classification-roc-f1-recall
'''

bank_related = bank.iloc[: , 7:11]

# %%
for i in bank_related.columns:
    if i!='duration':
        print(bank_related[i].value_counts())
        print('-----------------')


# %%
'''duration'''
fig, (ax1, ax2) = plt.subplots(nrows = 1, ncols = 2, figsize = (13, 5))
sns.boxplot(x = 'duration', data = bank_related, orient = 'v', ax = ax1)
ax1.set_xlabel('Calls', fontsize=10)
ax1.set_ylabel('Duration', fontsize=10)
ax1.set_title('Calls Distribution', fontsize=10)
ax1.tick_params(labelsize=10)

sns.distplot(bank_related['duration'], ax = ax2)
sns.despine(ax = ax2)
ax2.set_xlabel('Duration Calls', fontsize=10)
ax2.set_ylabel('Occurence', fontsize=10)
ax2.set_title('Duration x Ocucurence', fontsize=10)
ax2.tick_params(labelsize=10)

plt.subplots_adjust(wspace=0.5)
plt.tight_layout() 

# %%
#duration in minute
bank_related.duration=bank_related.duration/60
duration_stat=round(bank_related.duration.describe(),2)
duration_IQR=duration_stat[6]-duration_stat[4]
upper_level=duration_stat[6]+1.5*duration_IQR
lower_level=duration_stat[4]-1.5*duration_IQR
outlier_prec=sum(bank_related.duration>upper_level)/bank_related.shape[0]
print('Outliers are:', round(outlier_prec,2), '%')
# %%
#draw 3 bank_related categorical into chart
fig, (ax1, ax2, ax3) = plt.subplots(nrows = 1, ncols = 3, figsize = (15,6))
sns.countplot(bank_related['contact'], ax = ax1)
ax1.set_xlabel('Contact', fontsize = 10)
ax1.set_ylabel('Count', fontsize = 10)
ax1.set_title('Contact Counts')
ax1.tick_params(labelsize=10)

sns.countplot(bank_related['month'], ax = ax2, order = ['mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'])
ax2.set_xlabel('Months', fontsize = 10)
ax2.set_ylabel('')
ax2.set_title('Months Counts')
ax2.tick_params(labelsize=10)

sns.countplot(bank_related['day_of_week'], ax = ax3)
ax3.set_xlabel('Day of Week', fontsize = 10)
ax3.set_ylabel('')
ax3.set_title('Day of Week Counts')
ax3.tick_params(labelsize=10)

plt.subplots_adjust(wspace=0.25)

# %%
from sklearn.preprocessing import LabelEncoder
labelencoder_X = LabelEncoder()
bank_related['contact']     = labelencoder_X.fit_transform(bank_related['contact']) 
bank_related['month']       = labelencoder_X.fit_transform(bank_related['month']) 
bank_related['day_of_week'] = labelencoder_X.fit_transform(bank_related['day_of_week']) 


# %%
#bin duration in to larger groups
#base on self-made standard
def duration(data):

    data.loc[data['duration'] <= 102/60, 'duration'] = 1
    data.loc[(data['duration'] > 102/60) & (data['duration'] <= 180/60)  , 'duration']    = 2
    data.loc[(data['duration'] > 180/60) & (data['duration'] <= 319/60)  , 'duration']   = 3
    data.loc[(data['duration'] > 319/60) & (data['duration'] <= 644.5/60), 'duration'] = 4
    data.loc[data['duration']  > 644.5/60, 'duration'] = 5

    return data
duration(bank_related)

# %%
#social and eco attribute
bank_se = bank.loc[: , ['emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed']]
bank_se.head()

# %%
#other attribute
bank_o = bank.loc[: , ['campaign', 'pdays','previous', 'poutcome']]
bank_o.head()

# %%
bank_o['poutcome'].unique()
bank_o['poutcome'].replace(['nonexistent', 'failure', 'success'], [1,2,3], inplace  = True)

# %%
'''join all 3 df parts together'''
bank_final= pd.concat([bank_client, bank_related, bank_se, bank_o], axis = 1)
bank_final = bank_final[['age', 'job', 'marital', 'education', 'default', 'housing', 'loan',
                     'contact', 'month', 'day_of_week', 'duration', 'emp.var.rate', 'cons.price.idx', 
                     'cons.conf.idx', 'euribor3m', 'nr.employed', 'campaign', 'pdays', 'previous', 'poutcome']]
bank_final.shape

# %%
       '''...Model Time...'''

'''build logistic regression model'''


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(bank_final, y, test_size = 0.2, random_state = 101)

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score
k_fold = KFold(n_splits=10, shuffle=True, random_state=0)

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
#%%

from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression() 
logmodel.fit(X_train,y_train)
logpred = logmodel.predict(X_test)


print(confusion_matrix(y_test, logpred))
print(round(accuracy_score(y_test, logpred),2)*100)
LOGCV = (cross_val_score(logmodel, X_train, y_train, cv=10, scoring = 'accuracy').mean())


# %%
'''build KNN model'''
from sklearn import model_selection
from sklearn.neighbors import KNeighborsClassifier

X_trainK, X_testK, y_trainK, y_testK = train_test_split(bank_final, y, test_size = 0.2, random_state = 101)

#Neighbors
neighbors = np.arange(0,25)

#Create empty list that will hold cv scores
cv_scores = []

#Perform 10-fold cross validation on training set for odd values of k:
for k in neighbors:
    k_value = k+1
    knn = KNeighborsClassifier(n_neighbors = k_value, weights='uniform', p=2, metric='euclidean')
    kfold = model_selection.KFold(n_splits=10, random_state=123)
    scores = model_selection.cross_val_score(knn, X_trainK, y_trainK, cv=kfold, scoring='accuracy')
    cv_scores.append(scores.mean()*100)
    print("k=%d %0.2f (+/- %0.2f)" % (k_value, scores.mean()*100, scores.std()*100))

optimal_k = neighbors[cv_scores.index(max(cv_scores))]
print ("The optimal number of neighbors is %d with %0.1f%%" % (optimal_k, cv_scores[optimal_k]))

plt.plot(neighbors, cv_scores)
plt.xlabel('Number of Neighbors K')
plt.ylabel('Train Accuracy')
plt.show()
#%%
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=22)
knn.fit(X_train, y_train)
knnpred = knn.predict(X_test)

print(confusion_matrix(y_test, knnpred))
print(round(accuracy_score(y_test, knnpred),2)*100)
KNNCV = (cross_val_score(knn, X_train, y_train, cv=k_fold, n_jobs=1, scoring = 'accuracy').mean())

#%%
'''build SVC model'''
from sklearn.svm import SVC
svc= SVC(kernel = 'sigmoid')
svc.fit(X_train, y_train)
svcpred = svc.predict(X_test)
print(confusion_matrix(y_test, svcpred))
print(round(accuracy_score(y_test, svcpred),2)*100)
SVCCV = (cross_val_score(svc, X_train, y_train, cv=k_fold, n_jobs=1, scoring = 'accuracy').mean())

#%%
'''build decision tree model'''
from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier(criterion='gini') #criterion = entopy, gini
dtree.fit(X_train, y_train)
dtreepred = dtree.predict(X_test)

print(confusion_matrix(y_test, dtreepred))
print(round(accuracy_score(y_test, dtreepred),2)*100)
DTREECV = (cross_val_score(dtree, X_train, y_train, cv=k_fold, n_jobs=1, scoring = 'accuracy').mean())

#%%
'''build randomforest model'''
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators = 200)#criterion = entopy,gini
rfc.fit(X_train, y_train)
rfcpred = rfc.predict(X_test)

print(confusion_matrix(y_test, rfcpred ))
print(round(accuracy_score(y_test, rfcpred),2)*100)
RFCCV = (cross_val_score(rfc, X_train, y_train, cv=k_fold, n_jobs=1, scoring = 'accuracy').mean()) 

#%%
'''gaussianNB'''
from sklearn.naive_bayes import GaussianNB
gaussiannb= GaussianNB()
gaussiannb.fit(X_train, y_train)
gaussiannbpred = gaussiannb.predict(X_test)
probs = gaussiannb.predict(X_test)

print(confusion_matrix(y_test, gaussiannbpred ))
print(round(accuracy_score(y_test, gaussiannbpred),2)*100)
GAUSIAN = (cross_val_score(gaussiannb, X_train, y_train, cv=k_fold, n_jobs=1, scoring = 'accuracy').mean())


#%%
'''XGB'''
'''
from xgboost import XGBClassifier
xgb = XGBClassifier()
xgb.fit(X_train, y_train)
xgbprd = xgb.predict(X_test)

print(confusion_matrix(y_test, xgbprd ))
print(round(accuracy_score(y_test, xgbprd),2)*100)
XGB = (cross_val_score(estimator = xgb, X = X_train, y = y_train, cv = 10).mean())
'''

#%%
from sklearn.ensemble import GradientBoostingClassifier
gbk = GradientBoostingClassifier()
gbk.fit(X_train, y_train)
gbkpred = gbk.predict(X_test)
print(confusion_matrix(y_test, gbkpred ))
print(round(accuracy_score(y_test, gbkpred),2)*100)
GBKCV = (cross_val_score(gbk, X_train, y_train, cv=k_fold, n_jobs=1, scoring = 'accuracy').mean())
#%%
'''grad boosting'''
from sklearn.ensemble import GradientBoostingClassifier
gbk = GradientBoostingClassifier()
gbk.fit(X_train, y_train)
gbkpred = gbk.predict(X_test)
print(confusion_matrix(y_test, gbkpred ))
print(round(accuracy_score(y_test, gbkpred),2)*100)
GBKCV = (cross_val_score(gbk, X_train, y_train, cv=k_fold, n_jobs=1, scoring = 'accuracy').mean())



#%%

'''compare models'''
models = pd.DataFrame({
                'Models': ['Random Forest Classifier', 'Decision Tree Classifier', 'Support Vector Machine',
                           'K-Near Neighbors', 'Logistic Model', 'Gausian NB', 'Gradient Boosting'],
                'Score':  [RFCCV, DTREECV, SVCCV, KNNCV, LOGCV, GAUSIAN, GBKCV]})

models.sort_values(by='Score', ascending=False)

# %%


'''        
                            analysis model's result
Accuracy is measured by the area under the ROC curve. 
An area of 1 represents a perfect test; an area of .5 represents a worthless test.
A rough guide for classifying the accuracy of a diagnostic test is the traditional 
academic point system:

.90-1 = excellent (A)

.80-.90 = good (B)

.70-.80 = fair (C)

.60-.70 = poor (D)

.50-.60 = fail (F)'''
#%%

from sklearn import metrics
#fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(nrows = 2, ncols = 3, figsize = (15, 4))
fig, ax_arr = plt.subplots(nrows = 2, ncols = 3, figsize = (20,15))

#LOGMODEL
probs = logmodel.predict_proba(X_test)
preds = probs[:,1]
fprlog, tprlog, thresholdlog = metrics.roc_curve(y_test, preds)
roc_auclog = metrics.auc(fprlog, tprlog)

ax_arr[0,0].plot(fprlog, tprlog, 'b', label = 'AUC = %0.2f' % roc_auclog)
ax_arr[0,0].plot([0, 1], [0, 1],'r--')
ax_arr[0,0].set_title('Receiver Operating Characteristic Logistic ',fontsize=20)
ax_arr[0,0].set_ylabel('True Positive Rate',fontsize=20)
ax_arr[0,0].set_xlabel('False Positive Rate',fontsize=15)
ax_arr[0,0].legend(loc = 'lower right', prop={'size': 16})

#RANDOM FOREST --------------------
probs = rfc.predict_proba(X_test)
preds = probs[:,1]
fprrfc, tprrfc, thresholdrfc = metrics.roc_curve(y_test, preds)
roc_aucrfc = metrics.auc(fprrfc, tprrfc)

ax_arr[0,1].plot(fprrfc, tprrfc, 'b', label = 'AUC = %0.2f' % roc_aucrfc)
ax_arr[0,1].plot([0, 1], [0, 1],'r--')
ax_arr[0,1].set_title('Receiver Operating Characteristic Random Forest ',fontsize=20)
ax_arr[0,1].set_ylabel('True Positive Rate',fontsize=20)
ax_arr[0,1].set_xlabel('False Positive Rate',fontsize=15)
ax_arr[0,1].legend(loc = 'lower right', prop={'size': 16})

#KNN----------------------
probs = knn.predict_proba(X_test)
preds = probs[:,1]
fprknn, tprknn, thresholdknn = metrics.roc_curve(y_test, preds)
roc_aucknn = metrics.auc(fprknn, tprknn)

ax_arr[0,2].plot(fprknn, tprknn, 'b', label = 'AUC = %0.2f' % roc_aucknn)
ax_arr[0,2].plot([0, 1], [0, 1],'r--')
ax_arr[0,2].set_title('Receiver Operating Characteristic KNN ',fontsize=20)
ax_arr[0,2].set_ylabel('True Positive Rate',fontsize=20)
ax_arr[0,2].set_xlabel('False Positive Rate',fontsize=15)
ax_arr[0,2].legend(loc = 'lower right', prop={'size': 16})

#DECISION TREE ---------------------
probs = dtree.predict_proba(X_test)
preds = probs[:,1]
fprdtree, tprdtree, thresholddtree = metrics.roc_curve(y_test, preds)
roc_aucdtree = metrics.auc(fprdtree, tprdtree)

ax_arr[1,0].plot(fprdtree, tprdtree, 'b', label = 'AUC = %0.2f' % roc_aucdtree)
ax_arr[1,0].plot([0, 1], [0, 1],'r--')
ax_arr[1,0].set_title('Receiver Operating Characteristic Decision Tree ',fontsize=20)
ax_arr[1,0].set_ylabel('True Positive Rate',fontsize=20)
ax_arr[1,0].set_xlabel('False Positive Rate',fontsize=15)
ax_arr[1,0].legend(loc = 'lower right', prop={'size': 16})

#GAUSSIAN ---------------------
probs = gaussiannb.predict_proba(X_test)
preds = probs[:,1]
fprgau, tprgau, thresholdgau = metrics.roc_curve(y_test, preds)
roc_aucgau = metrics.auc(fprgau, tprgau)

ax_arr[1,1].plot(fprgau, tprgau, 'b', label = 'AUC = %0.2f' % roc_aucgau)
ax_arr[1,1].plot([0, 1], [0, 1],'r--')
ax_arr[1,1].set_title('Receiver Operating Characteristic Gaussian ',fontsize=20)
ax_arr[1,1].set_ylabel('True Positive Rate',fontsize=20)
ax_arr[1,1].set_xlabel('False Positive Rate',fontsize=15)
ax_arr[1,1].legend(loc = 'lower right', prop={'size': 16})

#ALL PLOTS ----------------------------------
ax_arr[1,2].plot(fprgau, tprgau, 'b', label = 'Gaussian', color='black')
ax_arr[1,2].plot(fprdtree, tprdtree, 'b', label = 'Decision Tree', color='blue')
ax_arr[1,2].plot(fprknn, tprknn, 'b', label = 'Knn', color='brown')
ax_arr[1,2].plot(fprrfc, tprrfc, 'b', label = 'Random Forest', color='green')
ax_arr[1,2].plot(fprlog, tprlog, 'b', label = 'Logistic', color='grey')
ax_arr[1,2].set_title('Receiver Operating Comparison ',fontsize=20)
ax_arr[1,2].set_ylabel('True Positive Rate',fontsize=20)
ax_arr[1,2].set_xlabel('False Positive Rate',fontsize=15)
ax_arr[1,2].legend(loc = 'lower right', prop={'size': 16})

plt.subplots_adjust(wspace=0.2)
plt.tight_layout() 

# %%


'''after finish ROC/AUC, select model base on recall and precision


ANALYZING THE RESULTS
So now we have to decide which one is the best model, and we have two types of wrong values:

False Positive,type 1 error, means the client do NOT SUBSCRIBED to term deposit, but the model thinks he did.
False Negative,type 2 error, means the client SUBSCRIBED to term deposit, but the model said he dont.
In my opinion:

The first one its most harmful, because we think that we already have that client but we dont 
and maybe we lost him in other future campaings.The second its not good but its ok, we have that 
client and in the future we'll discovery that in truth he's already our client
So, our objective here, is to find the best model by confusion matrix with the lowest False 
Positive as possible.Obs1 - lets go back and look the best confusion matrix that attend 
this criteria Obs2 - i'll do the math manualy to be more visible and understanding

'''
from sklearn.metrics import classification_report
print('KNN Confusion Matrix\n', confusion_matrix(y_test, knnpred))
print('KNN Reports\n',classification_report(y_test, knnpred))

# %%

print('GaussianBN Confusion Matrix\n', confusion_matrix(y_test, gaussiannbpred))
print('GaussianBN Reports\n',classification_report(y_test, gaussiannbpred))

# %%
