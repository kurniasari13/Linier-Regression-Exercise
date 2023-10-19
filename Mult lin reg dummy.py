import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import statsmodels.api as sm 
import seaborn as sns
sns.set()

raw_data = pd.read_csv('D:/DATA ANALYST/belajar_python/LATIHAN DS/29588090_1.03.Dummies.csv')

data = raw_data.copy()
data['Attendance'] = data['Attendance'].map({'Yes':1 ,'No': 0})
#print(data.describe())

y = data['GPA']
x1 = data[['SAT','Attendance']]
x = sm.add_constant(x1)
results = sm.OLS(y,x).fit()
#print(results.summary())

plt.scatter(data['SAT'], y,c=data['Attendance'],cmap='RdYlGn_r')
yhat_no = 0.6439 + 0.0014*data['SAT']
yhat_yes = 0.8665 + 0.0014*data['SAT']
yhat = 0.275 + 0.0017*data['SAT']
fig = plt.plot(data['SAT'],yhat_no,c='orange', lw=3)
fig = plt.plot(data['SAT'],yhat_yes,c='red', lw=2)
fig = plt.plot(data['SAT'],yhat, lw = 4, c = 'blue')
plt.xlabel('SAT')
plt.ylabel('GPA')
#plt.show()

print(x.head())

data_new = pd.DataFrame({'const':1, 'SAT':[1700,1670], 'Attendance':[0,1]})
data_new = data_new[['const','SAT','Attendance']]
print((data_new.rename(index={0:'Bob',1:'Alice'})).head())
#print(data_new.head())

predictions = results.predict(data_new)
print(predictions)

predictionsdf = pd.DataFrame({'Predictions':predictions})
joined = data_new.join(predictionsdf)
print((joined.rename(index={0:'Bob',1:'Alice'})).head())
#print(joined.head())


