import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import statsmodels.api as sm 
import seaborn as sns
sns.set()
from sklearn.linear_model import LinearRegression

data = pd.read_csv('D:/DATA ANALYST/belajar_python/LATIHAN DS/29588022 real estate price size.csv')

#USING STATMODELS
#print(data.head())
#print(data.describe())

#y = data['price']
#x1 = data['size']

#plt.scatter(x1, y)
#plt.show()

#x = sm.add_constant(x1)
#results = sm.OLS(y,x).fit()
#print(results.summary())

#plt.scatter(x1,y)
#yhat = x1*223.1787+101900
#fig = plt.plot(x1,yhat, lw=4, c='orange', label ='regression line')
#plt.xlabel('Size', fontsize = 20)
#plt.ylabel('Price', fontsize = 20)
#plt.show()


#USING SKLEARN

y = data['price']
x = data['size']

plt.scatter(x, y)
plt.xlabel('size')
plt.ylabel('price')
plt.show()

x_matrix = x.values.reshape(-1,1)
reg = LinearRegression()
reg.fit(x_matrix, y)

print(reg.score(x_matrix, y))
print(reg.intercept_)
print(reg.coef_)

new_data = pd.DataFrame(data=[750,800],columns=['size'])
new_data['Predicted_price']= reg.predict(new_data)
print(new_data.head())

plt.scatter(x, y)
yhat = reg.intercept_ + reg.coef_*x_matrix
fig = plt.plot(x,yhat,c='red',lw=3)
plt.xlabel('size')
plt.ylabel('price')
plt.show()


