import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import statsmodels.api as sm 
import seaborn as sns
sns.set()
from sklearn.linear_model import LinearRegression

data = pd.read_csv('D:/DATA ANALYST/belajar_python/LATIHAN DS/29588058 1.02.Multiple linear regression.csv')

y = data['GPA']
x = data[['SAT','Rand 1,2,3']]

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(x)
x_scaled = scaler.transform(x)

reg = LinearRegression()
reg.fit(x_scaled, y)
#print(reg.intercept_)
#print(reg.coef_)

reg_summary = pd.DataFrame(data=[['Bias'],['SAT'],['Rand 1,2,3']],columns=['Features'])
reg_summary['Weight'] = reg.intercept_,reg.coef_[0],reg.coef_[1]
print(reg_summary)

new_data = pd.DataFrame(data=[[1700,2],[1800,1]],columns=[['SAT','Rand 1,2,3']])
new_data_scaled = scaler.transform(new_data)
print(reg.predict(new_data_scaled))

