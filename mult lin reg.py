import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import statsmodels.api as sm 
import seaborn as sns
sns.set()
from sklearn.linear_model import LinearRegression

data = pd.read_csv('D:/DATA ANALYST/belajar_python/LATIHAN DS/29588058 1.02.Multiple linear regression.csv')

#USING STATMODELS

#y = data['GPA']
#x1 = data[['SAT','Rand 1,2,3']]

#x = sm.add_constant(x1)
#results = sm.OLS(y,x).fit()
#print(results.summary())


#USING SKLEARN

y = data['GPA']
x = data[['SAT','Rand 1,2,3']]

reg = LinearRegression()
reg.fit(x, y)
print(reg.intercept_)
print(reg.coef_)
print(reg.score(x,y))
#print(x.shape)

r2 = reg.score(x, y)
n = x.shape[0]
p = x.shape[1]

adjusted_r2 = 1-(1-r2)*(n-1)/(n-p-1)
#print(adjusted_r2)

def adjr2(x,y):
    r2 = reg.score(x,y)
    n = x.shape[0]
    p = x.shape[1]
    adjusted_r2 = 1-(1-r2)*(n-1)/(n-p-1)
    return adjusted_r2

print(adjr2(x,y))

from sklearn.feature_selection import f_regression
p_values = f_regression(x, y)[1]
print(p_values)
print(p_values.round(3))

reg_summary = pd.DataFrame(data=x.columns.values, columns=['Features'])
reg_summary['Coefficients'] = reg.coef_
reg_summary['P-values'] = p_values.round(3)
print(reg_summary)
