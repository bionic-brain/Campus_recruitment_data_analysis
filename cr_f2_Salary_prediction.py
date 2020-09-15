# importing essential libraries
import pandas as pd
import numpy as np

# importing dataset
dataset = pd.read_csv("Campus_Recruitment_Data_Analysis/datasets_596958_1073629_Placement_Data_Full_Class.csv", index_col='sl_no')

# filtering non placed students
ds1 = dataset[dataset.status == 'Placed']
ds1 = ds1.drop(['status'], axis=1)
x = ds1.iloc[:, :-1].values
y = ds1.iloc[:, 12].values

# encoding of categorical variables
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
le_x = LabelEncoder()

x[:, 0] = le_x.fit_transform(x[:,0])
x[:, 2] = le_x.fit_transform(x[:,2])
x[:, 4] = le_x.fit_transform(x[:,4])
x[:, 5] = le_x.fit_transform(x[:,5])
x[:, 7] = le_x.fit_transform(x[:,7])
x[:, 8] = le_x.fit_transform(x[:,8])
x[:, 10] = le_x.fit_transform(x[:,10])

from sklearn.compose import ColumnTransformer
var_ct = ColumnTransformer([('encoder',OneHotEncoder(),[0,2,4,5,7,8,10])],remainder='passthrough')
x = np.array(var_ct.fit_transform(x))

# removing one variable from each feature to avoid dummy variable trap
x = x[:,[1,3,5,7,8,10,11,13,15,16,17,18,19,20]]
x = x.astype(np.float64)

# using backward elimination technique on ordinary least square method
# significance level for model is 0.05
import statsmodels.api as sm
newX = np.append(arr= np.ones((len(x),1)).astype(float), values=x, axis=1)
xopt = newX[:, :]
reg_o = sm.OLS(endog=y , exog=xopt).fit()
reg_o.summary()

xopt = newX[:, [0,1,2,3,4,5,6,7,8,9,10,11,12,13]]
reg_o = sm.OLS(endog=y , exog=xopt).fit()
reg_o.summary()

xopt = newX[:, [1,2,3,4,5,6,7,8,9,10,11,12,13]]
reg_o = sm.OLS(endog=y , exog=xopt).fit()
reg_o.summary()

from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(xopt,y, test_size=0.20, random_state=234)

# applying linear regression
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(xtrain,ytrain)

# prediction test set
ypred = reg.predict(xtest)

# graphical comparison of actual vs predicted salaries
import matplotlib.pyplot as plt
plt.plot(np.arange(len(ytest)), ytest, label='actual')
plt.plot(np.arange(len(ytest)), ypred, label='predicted')
plt.xlabel('Student ids (normalised)')
plt.ylabel('Salary')
plt.grid(True,ls = '-.', lw = 0.25)
plt.legend()
plt.show()

# performance evaluation
print(reg_o.rsquared, reg_o.rsquared_adj)