import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import statsmodels.api as sm 
import seaborn as sns
sns.set()

raw_data = pd.read_csv('D:/DATA ANALYST/belajar_python/LATIHAN DS/29588076 real estate price size year.csv')
x = raw_data[['size','year']]
y = raw_data['price']

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(x)
x_scaled = scaler.transform(x)

from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(x_scaled, y)
print(reg.coef_)
print(reg.intercept_)

r2 = reg.score(x_scaled, y)
print(r2)

def adjr2(x_scaled,y):
    r2 = reg.score(x_scaled, y)
    n = x.shape[0]
    p = x.shape[1]
    adjusted_r2 = 1-(1-r2)*(n-1)/(n-p-1)
    return adjusted_r2
print(adjr2(x_scaled, y))

new_data = pd.DataFrame(data=[[750,2009]],columns=[['size','year']])
new_data_scaled = scaler.transform(new_data)
print(reg.predict(new_data_scaled))

from sklearn.feature_selection import f_regression
p_values = f_regression(x_scaled, y)[1]
print(p_values.round(3))

summary = pd.DataFrame(data=['Bias','Size','Year'],columns=['Features'])
summary['Weight'] = reg.intercept_ , reg.coef_[0], reg.coef_[1]
summary['P-values'] = 'N/A', p_values.round(3)[0], p_values.round(3)[1]
print(summary)
