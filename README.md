# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import libraries (NumPy, pandas, Matplotlib).
2. Load dataset (Startup.csv).
3. Extract X (R&D Spend) and y (Profit).
4. Normalize X values.
5. Initialize parameters (m = 0, b = 0).
6. Set learning rate and number of epochs.
7. Loop for epochs:
8. Predict y using y_pred = mX + b
9. Compute gradients (dm, db)
10. Update m and b
11. Print final slope (m) and intercept (b).
12. Plot scatter points and regression line.

## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: cholimgapuram sai likitha
RegisterNumber: 212224230046
/*
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
def linear_regression(X1, y, learning_rate=0.1, num_iters=1000):
    X = np.c_[np.ones(len(X1)), X1]
    theta = np.zeros((X.shape[1], 1))
    m = len(X1)
    for _ in range(num_iters):
        predictions = X.dot(theta)
        errors = predictions - y
        theta -= learning_rate * (1 / m) * X.T.dot(errors)
    return theta
data = pd.read_csv("50_Startups (1).csv")
print("Dataset:")
print(data.head())
X = data.iloc[:, :-2].values.astype(float)
y = data.iloc[:, -1].values.reshape(-1, 1)
X_scaler = StandardScaler()
y_scaler = StandardScaler()
X_scaled = X_scaler.fit_transform(X)
y_scaled = y_scaler.fit_transform(y)
print("\nScaled Features:")
print(X_scaled)
theta = linear_regression(X_scaled, y_scaled)
print("\nTheta Values:")
print(theta)
new_data = np.array([[165349.2, 136897.8, 471784.1]])
new_scaled = X_scaler.transform(new_data)
new_scaled = np.c_[np.ones(len(new_scaled)), new_scaled]
prediction_scaled = new_scaled.dot(theta)
prediction = y_scaler.inverse_transform(prediction_scaled)
print("\nScaled Prediction:")
print(prediction_scaled)
print("\nPredicted Profit:")
print(prediction[0][0])

```

## Output:
<img width="477" height="1023" alt="image" src="https://github.com/user-attachments/assets/d9c0d7e4-4429-4142-8a26-a5ce84e276eb" />




## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
