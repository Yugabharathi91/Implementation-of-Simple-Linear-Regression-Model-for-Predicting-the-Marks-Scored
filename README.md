# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
 step 1.start step 2.Import the standard Libraries.

step 3.Set variables for assigning dataset values.

step 4.Import linear regression from sklearn.

step 5.Assign the points for representing in the graph.

step 6.Predict the regression for marks by using the representation of the graph.

step 7.Compare the graphs and hence we obtained the linear regression for the given datas.

step 8.stop



## Program:
```
 /*
Program to implement the simple linear regression model for predicting the marks scored.

Developed by: MOHAMED SULTHAN A 

RegisterNumber: 212223230125

*/


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv(r"C:\Users\admin\Downloads\student_scores.csv")
print(df)
df.head(0)
df.tail(0)
print(df.head())
print(df.tail())
x = df.iloc[:,:-1].values
print(x)
y = df.iloc[:,1].values
print(y)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
print(y_pred)
print(y_test)
#Graph plot for training data
plt.scatter(x_train,y_train,color='black')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
#Graph plot for test data
plt.scatter(x_test,y_test,color='black')
plt.plot(x_train,regressor.predict(x_train),color='red')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_absolute_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE= ",rmse)
```

## Output:
 ## DATA SET:
 ![image](https://github.com/user-attachments/assets/097c6633-5a11-4e6c-86f8-83d1b12cec5e)
 ## HARD VALUES:
 ![image](https://github.com/user-attachments/assets/255617cb-5dc6-4ae7-9cf3-8b59ded815b7)
 ## TAIL VALUES:
 ![image](https://github.com/user-attachments/assets/245a0944-7776-4681-b6f9-cb4e79215f29)
 ## X AND Y VALUES:
 ![image](https://github.com/user-attachments/assets/b94cfb6f-9a2c-4793-a722-85e5f172f4f8)
 PREDICTION OF X AND Y:
 ![image](https://github.com/user-attachments/assets/0548cd88-7cb4-4753-b544-820760f0089b)
## MSE,MAE, AMD RMSE:
![image](https://github.com/user-attachments/assets/cfb97d11-f8a9-4553-9d81-c8a7ca52f4e4)
## TRAINING SET:
![image](https://github.com/user-attachments/assets/3832e7a6-0337-48c1-ac11-c20e1e610f53)
![image](https://github.com/user-attachments/assets/73cfff45-fadb-4a3d-bd30-1c4938a65496)









## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
