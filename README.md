# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the standard libraries.
2. Upload the dataset and check for any null values using .isnull() function.
3. Import LabelEncoder and encode the dataset.
4. Import DecisionTreeRegressor from sklearn and apply the model on the dataset.
5. Predict the values of arrays.
6. Import metrics from sklearn and calculate the MSE and R2 of the model on the dataset.
7. Predict the values of array.
8. Apply to new unknown values.

## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: Roop Sagar S L
RegisterNumber:  212223040175
*/

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
data = pd.read_csv("Salary_EX7.csv")
data.head()
data.info()
data.isnull().sum()
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data["Position"] = le.fit_transform(data["Position"])
data.head()
x=data[["Position","Level"]]
y=data["Salary"]
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=2)
from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
from sklearn import metrics
mse = metrics.mean_squared_error(y_test,y_pred)
mse
r2=metrics.r2_score(y_test,y_pred)
r2
dt.predict([[5,6]])
plt.figure(figsize=(20, 8))
plot_tree(dt, feature_names=x.columns, filled=True)
plt.show()

```

## Output:

## HEAD(),INFO() & NULL():

![head-info-null](https://github.com/user-attachments/assets/d9a62958-6f4b-4fbb-8c7a-c08254a29569)

## Converting string literals to numerical values using label encoder:

![Converting-string-literals](https://github.com/user-attachments/assets/9ef2a632-94dd-4be0-a7ba-fee2ae063522)

## MEAN SQUARED ERROR:

![mean-sq-error](https://github.com/user-attachments/assets/5e87fbd4-6833-453b-96b4-ebf44c5dc64b)

## R2 (VARIANCE):

![r2](https://github.com/user-attachments/assets/47eb4a4b-14f3-40ed-b69b-d2e942471819)

## DATA PREDICTION & DECISION TREE REGRESSOR FOR PREDICTING THE SALARY OF THE EMPLOYEE:

![data-prediction](https://github.com/user-attachments/assets/2435df53-34be-463e-bdc6-2f4259286c81)


## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
