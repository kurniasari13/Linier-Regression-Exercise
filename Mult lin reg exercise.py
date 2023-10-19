import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import statsmodels.api as sm 
import seaborn as sns
sns.set()

raw_data = pd.read_csv('D:/DATA ANALYST/belajar_python/LATIHAN DS/29588076 real estate price size year.csv')


#USING STATMODELS

#y = raw_data['price']
#x1 = raw_data[['size','year']]
#print(raw_data.head())

#x = sm.add_constant(x1)
#results = sm.OLS(y,x).fit()
#print(results.summary())

#plt.scatter(raw_data['size'], y)
#yhat = -5772000 + 227.7009*raw_data['size'] + 2916.7853*raw_data['year']
#fig= plt.plot(raw_data['size'],yhat, c='red', lw=2)
#plt.show()


#USING SKLEARN
x = raw_data[['size','year']]
y = raw_data['price']

from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(x, y)
print(reg.coef_)
print(reg.intercept_)

r2 = reg.score(x, y)
print(r2)

def adjr22(x,y):
    r2 = reg.score(x, y)
    n = x.shape[0]
    p = x.shape[1]
    adjusted_r2 = 1-(1-r2)*(n-1)/(n-p-1)
    return adjusted_r2
print(adjr22(x, y))

new_data = pd.DataFrame(data=[[750,2009]],columns=[['size','year']])
print(reg.predict(new_data))

from sklearn.feature_selection import f_regression
p_values = f_regression(x, y)[1]
print(p_values.round(3))

summary = pd.DataFrame(data=['Bias','Size','Year'],columns=['Features'])
summary['Weight'] = reg.intercept_ , reg.coef_[0], reg.coef_[1]
summary['P-values'] = 'N/A', p_values.round(3)[0], p_values.round(3)[1]
print(summary)
