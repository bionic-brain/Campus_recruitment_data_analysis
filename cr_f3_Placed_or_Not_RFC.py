import pandas as pd
import numpy as np
dataset = pd.read_csv("Campus_Recruitment_Data_Analysis/datasets_596958_1073629_Placement_Data_Full_Class.csv", index_col='sl_no')

x = dataset.iloc[:,:-2].values
y = dataset.iloc[:,12].values

from sklearn.preprocessing import LabelEncoder,OneHotEncoder
le_x = LabelEncoder()
le_y = LabelEncoder()
y = le_y.fit_transform(y)

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


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.20, random_state = 234)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# Training the Random Forest Classification model on the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 432)
classifier.fit(X_train, y_train)


# Predicting the Test set results
y_pred = classifier.predict(X_test)


from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
result = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(result)
result1 = classification_report(y_test, y_pred)
print("Classification Report:",)
print (result1)
result2 = accuracy_score(y_test,y_pred)
print("Accuracy:",result2)
