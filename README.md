# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Gather data consisting of two variables. Input- a factor that affects the marks and Output - the marks scored by students
2. Plot the data points on a graph where x-axis represents the input variable and y-axis represents the marks scored
3. Define and initialize the parameters for regression model: slope controls the steepness and intercept represents where the line crsses the y-axis
4. Use the linear equation to predict marks based on the input Predicted Marks = m.(hours studied) + b
for each data point calculate the difference between the actual and predicted marks
5. Adjust the values of m and b to reduce the overall error. The gradient descent algorithm helps update these parameters based on the calculated error
7. Once the model parameters are optimized, use the final equation to predict marks for any new input data



## Program:
Developed by:   YUGABHARATHI M
RegisterNumber:  212224230314
```
/*
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#import libraries to find mae, mse
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

#read csv file
df= pd.read_csv('data.csv')

#displaying the content in datafile
df.head()
df.tail()

# Segregating data to variables
X=df.iloc[:,:-1].values
X
y=df.iloc[:,-1].values
y

#splitting train and test data
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=1/2,random_state=0)

#import linear regression model and fit the model with the data
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,y_train)

#displaying predicted values
y_pred=regressor.predict(X_test)
y_pred

#displaying actual values
y_test

#graph plot for training data
import matplotlib.pyplot as plt
plt.scatter(X_train,y_train,color='red')
plt.plot(X_train,regressor.predict(X_train),color='blue')
plt.title("Hours vs Scores (Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")

#graph plot for test data
plt.scatter(X_test,y_test,color='red')
plt.plot(X_test,regressor.predict(X_test),color='blue')
plt.title("Hours vs Scores (Testing Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")

#find mae,mse,rmse
mse=mean_squared_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print('RMSE = ',rmse)
*/
```

## Output:

## head values:
<img width="167" height="241" alt="image" src="https://github.com/user-attachments/assets/5201bb30-6a77-4600-b17d-0bf5a5b9d2fc" />

## Tail values:
<img width="178" height="235" alt="image" src="https://github.com/user-attachments/assets/8ad4ae9b-45d9-4e1d-aad5-591cfa8fdf49" />

## x values:
<img width="160" height="557" alt="image" src="https://github.com/user-attachments/assets/d95f187c-9581-4466-8aba-bdf09b414df4" />

## y values:
<img width="718" height="60" alt="image" src="https://github.com/user-attachments/assets/97e7854e-1e7d-4187-b320-35df18233f65" />

## predicted values:
<img width="698" height="74" alt="image" src="https://github.com/user-attachments/assets/2f12c25c-6cd6-4a4e-b0d1-118d4517e1d7" />

## actual values:
<img width="576" height="28" alt="image" src="https://github.com/user-attachments/assets/51c60503-cb98-4234-abe3-4e1fc78eb3a6" />

## Training set:
<img width="562" height="455" alt="image" src="https://github.com/user-attachments/assets/935e65ae-77b9-423b-b0fd-1e34502a2aaf" />

## Testing set:
<img width="562" height="455" alt="image" src="https://github.com/user-attachments/assets/ef50ae93-a2e4-4049-94d3-de9ae2d53294" />

 ## MSE,MAE and RSME:
 <img width="258" height="66" alt="image" src="https://github.com/user-attachments/assets/6828e787-b6f3-4aa7-b7f2-128b40b78c65" />





## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
