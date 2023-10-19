import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import statsmodels.api as sm 
import seaborn as sns
sns.set()

raw_data = pd.read_csv('D:/DATA ANALYST/belajar_python/LATIHAN DS/29588130 real estate price size year view.csv')

data = raw_data.copy()
data['view'] = data['view'].map({'Sea view':1,'No sea view':0})

y = data['price']
x1 = data[['size','year','view']]

x = sm.add_constant(x1)
results = sm.OLS(y,x).fit()
print(results.summary())

plt.scatter(data['size'], y,c=data['view'],cmap='RdYlGn_r')
yhat_no = -5398000 + 223.0316*data['size'] + 2718.9489*data['year']
yhat_yes = -5341270 + 223.0316*data['size'] + 2718.9489*data['year']
fig = plt.plot(data['size'],yhat_no,c='orange', lw=2)
fig = plt.plot(data['size'],yhat_yes,c='red', lw=2)
plt.xlabel('size', fontsize=20)
plt.ylabel('price', fontsize=20)
plt.show()
