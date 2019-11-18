import os
os.getcwd()
os.chdir("C:/Users/User/Documents/Python Scripts/Sample_Data/Data/titanic")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

Train_data = pd.read_csv("train.csv", low_memory = False)
Test_data = pd.read_csv("test.csv", low_memory = False)
Train_data["source"] = "TRAIN"
Test_data["Survived"] = 0
Test_data["source"] = "TEST"

Full_data  = pd.concat([Train_data, Test_data], axis = 0, sort = False)

Full_data.isnull().sum()
sns.heatmap(Full_data.isnull(), yticklabels = False, cbar = False, cmap = "viridis")

data  =Full_data.copy()
data["Cabin"] = data["Cabin"].fillna(0)

Full_data["Cabin"] = Full_data["Cabin"].fillna(0)
Full_data["Cabin"] = np.where(Full_data.Cabin == 0 ,0,1)

sns.pairplot(Full_data)
Full_data['Pclass'].hist(bins = 5, color = 'red', alpha = 1, palette = 'whitegrid')

# import cufflinks as cf

plt.figure(figsize=(12,7))
sns.boxplot(x ='Pclass', y ='Age', data = Full_data, palette = 'winter')
sns.boxplot(x='Sex', y = 'Age', data = Full_data, palette = 'winter')

def impute_age(cols):
    Age = cols[0]
    Pclass = cols[1]
   
    if pd.isnull(Age):

        if Pclass == 1:
            return 37

        elif Pclass == 2:
            return 29

        else:
            return 24

    else:
        return Age

Full_data['Age'] = Full_data[['Age','Pclass']].apply(impute_age,axis=1)

for Col_Name in Full_data.columns:
    if(Full_data[Col_Name].dtype == object):
        Impute_Value = Full_data.loc[Full_data["source"] == "TRAIN", Col_Name].mode()[0]
        Full_data[Col_Name] = Full_data[Col_Name].fillna(Impute_Value)
    else:
        Impute_Value = Full_data.loc[Full_data["source"] == "TRAIN", Col_Name].median()
        Full_data[Col_Name].fillna(Impute_Value, inplace = True)

Full_data.isnull().sum()


Full_data.describe()
Full_data.shape

Full_data['Fare'].quantile([0.10,0.20,0.90,0.92,0.96, 0.97, 0.98, 0.99] , interpolation  ='linear')
Full_data['Fare'].hist(bins = 5, color = 'red', alpha = 1)
#Since Feature 'Fare' has outliers on the upper bound so we are capping it at 96% i.e 146.52
Full_data['Fare'] = np.where(Full_data['Fare']>146.5208, 146.5208, Full_data['Fare'])

Full_data['Sex'] = np.where(Full_data['Sex'] == 'male', 1, 0)

Full_data['Pclass'].value_counts()
Full_data['SibSp'].value_counts() 
Full_data['Parch'].value_counts() 
Full_data['Embarked'].value_counts() # Dummy

# Dummy Variable
Dummy_Df = pd.get_dummies(Full_data['Embarked'])
Fulldata= pd.concat([Full_data, Dummy_Df], axis = 1)

#Drop  Name, embarked and Ticket No.

Fulldata = Fulldata.drop(['Name', 'Embarked', 'Ticket'], axis = 1)

Fulldata['Survived'].value_counts()

#Bifurcation & Dropping Source
Train = Fulldata.loc[Fulldata['source'] == 'TRAIN',:].drop('source', axis = 1).copy()
Test = Fulldata.loc[Fulldata['source'] == 'TEST',:].drop('source', axis = 1).copy()
Test = Test.drop(['Survived'], axis = 1) # Dropping Column Survived

#Further divide Train into Train X and Train Y
Train_X = Train.drop('Survived', axis = 1).copy()
Train_Y = Train['Survived'].copy()


# Check for VIF
from statsmodels.stats.outliers_influence import variance_inflation_factor

cols_to_drop_vif = []
for i in range(Train_X.shape[1]-1):
    temp_vif = variance_inflation_factor(Train_X.values, i) # Pass Train_X.values and i (col_number)
    print(Train_X.columns[i], ": ", temp_vif)
    if(temp_vif>10):
        print('Since vif value is greater than 10 so dropping the column ',Train_X.columns[i])
        cols_to_drop_vif.append(Train_X.columns[i])
    
Train_X.drop(cols_to_drop_vif, axis=1, inplace=True)
Test.drop(cols_to_drop_vif, axis=1, inplace=True)

## Correlation Check
Corr_Var = Train_X.corr()
Corr_Var.to_csv('Corr_Var.csv',header=True)

## Modelling
from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()
logmodel.fit(Train_X,Train_Y)

Prediction = logmodel.predict(Test)

Prediction = pd.DataFrame(Prediction)
Prediction.to_csv('Pred.csv',header=True)