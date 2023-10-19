import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression

raw_data1 = pd.read_csv('D:/DATA ANALYST/belajar_python/LATIHAN DS/MECHINE LEARNING/LINIER REG/29588446 1.04.Real life example.csv')
raw_data2 = raw_data1.copy()
#print(raw_data)
#print(data.describe(include='all'))
data = raw_data2.drop(['Model'],axis=1)
#print(raw_data)
#print(data.describe(include='all'))

data_no_mv = data.dropna(axis=0)
#print(data_no_mv.isnull().sum())
#print(data_no_mv.describe(include='all'))

#OUTLIERS VARIABEL PRICE
#sns.distplot(data_no_mv['Price'])
#plt.show()
q = data_no_mv['Price'].quantile(0.99)
data_1 = data_no_mv[data_no_mv['Price']<q]
#print(data_1.describe(include='all'))
#sns.distplot(data_1['Price'])
#plt.show()

#OUTLIERS VARIABEL MILEAGE
#sns.distplot(data_1['Mileage'])
#plt.show()
q = data_1['Mileage'].quantile(0.99)
data_2 = data_1[data_1['Mileage']<q]
#print(data_2.describe(include='all'))
#sns.distplot(data_2['Mileage'])
#plt.show()

#OUTLIERS VARIABEL ENGINEV
#sns.distplot(data_2['EngineV'])
#plt.show()
data_3 = data_2[data_2['EngineV']<6.5]
#print(data_3.describe(include='all'))
#sns.distplot(data_3['EngineV'])
#plt.show()

#OUTLIERS VARIABEL YEAR
#sns.distplot(data_3['Year'])
#plt.show()
q = data_3['Year'].quantile(0.01)
data_4 = data_3[data_3['Year']>q]
#print(data_4.describe(include='all'))
#sns.distplot(data_4['Year'])
#plt.show()

data_cleaned = data_4.reset_index(drop=True)
#print(data_cleaned)
#print(data_cleaned.describe(include='all'))\

#CEK LINEARITI VARIABEL
f, (ax1,ax2,ax3) = plt.subplots(1,3, sharey=True, figsize=(15,3))
ax1.scatter(data_cleaned['Mileage'],data_cleaned['Price'])
ax1.set_title('Price and Mileage')
ax2.scatter(data_cleaned['EngineV'],data_cleaned['Price'])
ax2.set_title('Price and EngineV')
ax3.scatter(data_cleaned['Year'],data_cleaned['Price'])
ax3.set_title('Price and Year')
plt.show()

sns.distplot(data_cleaned['Price'])
#plt.show()

#VARIABEL PRICE DI LOG TRANSFORM
Log_price = np.log(data_cleaned['Price'])
data_cleaned['Log_price'] = Log_price
#print(data_cleaned)

#CEK LINIERITI LAGI SETELAH LOG TRANSFORM
f, (ax1,ax2,ax3) = plt.subplots(1,3,sharey=True, figsize=(15,3))
ax1.scatter(data_cleaned['Mileage'],data_cleaned['Log_price'])
ax1.set_title('Log Price and Mileage')
ax2.scatter(data_cleaned['EngineV'],data_cleaned['Log_price'])
ax2.set_title('Log Price and EngineV')
ax3.scatter(data_cleaned['Year'],data_cleaned['Log_price'])
ax3.set_title('Log Price and Year')
plt.show()

data_cleaned = data_cleaned.drop(['Price'],axis=1)
#print(data_cleaned)
#print(data_cleaned.columns.values)

#CEK MULTIKOLINIERITI
from statsmodels.stats.outliers_influence import variance_inflation_factor
variables_numerical = data_cleaned[['Mileage','EngineV','Year']]
vif = pd.DataFrame()
vif['VIF'] = [variance_inflation_factor(variables_numerical.values, i) for i in range(variables_numerical.shape[1])]
vif['Features'] = variables_numerical.columns
#print(vif)

data_no_multicollinearity = data_cleaned.drop(['Year'],axis=1)
#print(data_no_multicollinearity.columns.values)

variabel = data_no_multicollinearity[['Mileage','EngineV']]
vif = pd.DataFrame()
vif['VIF'] = [variance_inflation_factor(variabel, i) for i in range(variabel.shape[1])]
vif['Features'] = variabel.columns
#print(vif)

#BIKIN VARIABEL DUMMY
data_with_dummies = pd.get_dummies(data_no_multicollinearity, drop_first=True)
#print(data_with_dummies)
#print(data_with_dummies.columns.values)
cols = ['Log_price','Mileage','EngineV',  'Brand_BMW','Brand_Mercedes-Benz', 'Brand_Mitsubishi', 'Brand_Renault','Brand_Toyota', 'Brand_Volkswagen',
 'Body_hatch' ,'Body_other' ,'Body_sedan' ,'Body_vagon' ,'Body_van','Engine Type_Gas', 'Engine Type_Other' ,'Engine Type_Petrol','Registration_yes']

data_preprocessed = data_with_dummies[cols]
#print((data_preprocessing.head()))

#BIKIN VARIABEL X DAN Y, LALU DI SCALING
targets = data_preprocessed['Log_price']
inputs = data_preprocessed.drop(['Log_price'], axis=1)
#print(inputs.columns.values)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(inputs)
inputs_scaled = scaler.transform(inputs)

#DATA DIBAGI JADI TRAIN AND TEST
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(inputs_scaled, targets, test_size=0.2, random_state= 365)

#REGRESI DAN BUAT SCATTER PLOT ANTARA Y TRAIN DAN Y HAT
reg = LinearRegression()
reg.fit(x_train, y_train)

y_hat = reg.predict(x_train)
plt.scatter(y_train, y_hat)
plt.xlabel('Targets (Y train)')
plt.ylabel('Prediction (Y Hat)')
plt.xlim(6,13)
plt.ylim(6,13)
plt.show()

#PLOT RESIDUAL
sns.distplot(y_train - y_hat)
plt.title('Residual PDF')
plt.show()

#SUMMARY
print(reg.score(x_train, y_train))
print(reg.intercept_)
weight_summary = pd.DataFrame(inputs.columns.values, columns=['Features'])
weight_summary['Weight'] = reg.coef_
print(weight_summary)
print(data_no_multicollinearity['Brand'].unique())
print(data_no_multicollinearity['Body'].unique())
print(data_no_multicollinearity['Engine Type'].unique())

#TESTING
y_hat_test = reg.predict(x_test)
plt.scatter(y_test, y_hat_test, alpha=0.2)
plt.xlabel('Targets (Y Test)')
plt.ylabel('Prediction (Y Hat Test)')
plt.xlim(6,13)
plt.ylim(6,13)
plt.show()

#SUMMARY TESTING
df_perform = pd.DataFrame(np.exp(y_hat_test), columns=['Predictions'])
y_test = y_test.reset_index(drop=True)
df_perform['Targets'] = np.exp(y_test)
df_perform['Residuals'] = df_perform['Targets'] - df_perform['Predictions']
df_perform['Differences%'] = np.absolute(df_perform['Residuals'] / df_perform['Targets']*100)
print(df_perform.describe())

pd.options.display.max_rows = 999
pd.set_option('display.float_format', lambda x: '%.2f' % x)
print(df_perform.sort_values(by=['Differences%']))
