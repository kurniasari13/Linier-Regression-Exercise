import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import statsmodels.api as sm 
import seaborn as sns
sns.set()
from sklearn.linear_model import LinearRegression


data = pd.read_csv('D:/DATA ANALYST/belajar_python/LATIHAN DS/MECHINE LEARNING/LINIER REG/29587970_1.01.Simple_linear_regression.csv')


#USING STATMODELS

#print(data.describe())
#y = data['GPA']
#x1 = data['SAT']

#plt.scatter(x1, y)
#plt.xlabel('SAT', fontsize = 20)
#plt.ylabel('GPA', fontsize = 20)
#plt.show()

#x = sm.add_constant(x1)
#print(x.head())
#print(x.describe())

#results = sm.OLS(y,x).fit()
#print(results.summary())

#plt.scatter(x1, y)
#yhat = 0.0017*x1 + 0.275
#fig = plt.plot(x1,yhat, lw=5, c='red', label ='regression line')
#plt.xlabel('SAT', fontsize = 20)
#plt.ylabel('GPA', fontsize = 20)
#plt.show()



#USING SKLEARN

x = data['SAT']
y = data['GPA']

#print(x.shape)
#print(y.shape)

x_matrix = x.values.reshape(-1,1)
#print(x_matrix.shape)

reg = LinearRegression()
reg.fit(x_matrix, y)
print(reg.score(x_matrix, y))
print(reg.coef_)
print(reg.intercept_)

new_data = pd.DataFrame(data=[1740,1760],columns=['SAT'])
#print(new_data.head())
print(reg.predict(new_data))

new_data['Predicted_GPA'] = reg.predict(new_data)
print(new_data.head())

plt.scatter(x,y)
yhat = reg.coef_*x_matrix + reg.intercept_
fig = plt.plot(x,yhat,lw=3,c='red')
plt.xlabel('SAT')
plt.ylabel('GPA')
plt.show()