# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Imports
2. Load dataset
3. Prepare input X and target Y
4. Train/test split
5. Create and train the linear model
6. Predict on test set
7. Plots
8. Evaluation metrics
9. Predict new samples
   
## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Imesh Karthick P R
RegisterNumber: 212225060092 
*/
```
import pandas as pd

import numpy as np 

import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error, mean_squared_error

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

df = pd.read_csv('exp_2_dataset_student_scores.csv')  

print("First 5 rows:\n", df.head(), "\n")

print("Last 5 rows:\n", df.tail(), "\n")

X = df.iloc[:, :-1].values   # all rows, all columns except last -> shape (n_samples, 1)

Y = df.iloc[:, -1].values    # all rows, last column -> shape (n_samples,)

print("X (features):", X.flatten())

print("Y (targets):", Y)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=1/3, random_state=0)

print("\nTraining samples:", len(X_train), " Testing samples:", len(X_test))

regressor = LinearRegression()

regressor.fit(X_train, Y_train)

Y_pred = regressor.predict(X_test)

print("\nPredicted values:", np.round(Y_pred, 2))

print("Actual values   :", Y_test)

plt.figure(figsize=(6,4))


plt.scatter(X_train, Y_train, color="orange", label="Training data")

plt.plot(X_train, regressor.predict(X_train), color="red", label="Fitted line")

plt.title("Hours vs Scores (Training set)")

plt.xlabel("Hours")

plt.ylabel("Scores")

plt.legend()

plt.grid(True)

plt.show()

order = np.argsort(X_test.flatten())

X_test_sorted = X_test.flatten()[order]

Y_test_sorted = Y_test[order]

Y_pred_sorted = Y_pred[order]

plt.figure(figsize=(6,4))

plt.scatter(X_test, Y_test, color="blue", label="Test data")

plt.plot(X_test_sorted, Y_pred_sorted, color="green", label="Predictions")

plt.title("Hours vs Scores (Testing set)")

plt.xlabel("Hours")

plt.ylabel("Scores")

plt.legend()

plt.grid(True)

plt.show()

mae = mean_absolute_error(Y_test, Y_pred)

mse = mean_squared_error(Y_test, Y_pred)

rmse = np.sqrt(mse)

print("\nMean Absolute Error (MAE):", mae)

print("Mean Squared Error (MSE):", mse)

print("Root Mean Squared Error (RMSE):", rmse)

new_hours = np.array([[2.5], [8.0]])   # shape must be (n_samples, 1)

pred_new = regressor.predict(new_hours)

print("\nPredictions for new hours", new_hours.flatten(), "=>", np.round(pred_new,2))



## Output:

<img width="596" height="383" alt="image" src="https://github.com/user-attachments/assets/dbaaae67-f2f7-4379-90f4-7627a950dd25" />

<img width="637" height="401" alt="image" src="https://github.com/user-attachments/assets/47055685-1a87-4c62-bf3d-5176a28e103c" />

<img width="413" height="71" alt="image" src="https://github.com/user-attachments/assets/5d038175-d333-4608-ad62-70a6b4be5a94" />


<img width="430" height="32" alt="image" src="https://github.com/user-attachments/assets/8b9f2db5-a625-451b-894d-5cabdcdbc3c0" />



## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
