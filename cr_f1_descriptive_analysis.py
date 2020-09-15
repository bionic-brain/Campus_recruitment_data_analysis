import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', 15)
pd.set_option('display.width', 400)
dataset = pd.read_csv("Campus_Recruitment_Data_Analysis/datasets_596958_1073629_Placement_Data_Full_Class.csv", index_col='sl_no')

dataset.describe()

# count plots
fig, axs = plt.subplots(ncols=4,figsize=(20,5))
sns.countplot(dataset['gender'], ax = axs[0])
sns.countplot(dataset['ssc_b'], ax = axs[1], palette="Paired")
sns.countplot(dataset['hsc_b'], ax = axs[2], palette="muted")
sns.countplot(dataset['hsc_s'], ax = axs[3], palette="dark")



# How much dependency between MBA percentage and Salary

plt.scatter(dataset['mba_p'].values, dataset['salary'].values)
plt.xlabel('MBA %')
plt.ylabel('Salary %')
plt.title('MBA percentage vs salary')
plt.show()

# salary distribution
plt.hist(dataset['salary'].values)
plt.xlabel('Salary %')
plt.ylabel('Count')
plt.title('Salary Distribution')
plt.show()

# MBA percentage distribution
plt.boxplot(dataset['mba_p'])
plt.ylabel('Mba %')
plt.title('MBA percentage Distribution')
plt.show()

# Ratio of students in each field of degree education
plt.pie(dataset['degree_t'].value_counts(),labels=dataset['degree_t'].unique(),autopct='%1.2f%%')
plt.title(" Ratio of students in each field of degree education")
plt.show()

# How much dependency between Degree percentage and Salary?
plt.scatter(dataset['degree_p'].values, dataset['salary'].values)
plt.xlabel('Degree %')
plt.ylabel('Salary')
plt.title('Degree percentage vs salary')
plt.show()

# Heatmap of co-relation matrix
plt.figure(figsize=(15,10))
corr = dataset.corr()
sns.heatmap(corr, annot = True)
