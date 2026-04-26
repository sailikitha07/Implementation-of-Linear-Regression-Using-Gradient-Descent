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
import matplotlib.pyplot as plt
data = pd.read_csv("Startup.csv")
X = data['R&D Spend'].values
y = data['Profit'].values
X = (X - X.mean()) / X.std()
m = 0
b = 0
learning_rate = 0.01
epochs = 1000
n = len(X)
for i in range(epochs):
    y_pred = m * X + b
    dm = (-2/n) * np.sum(X * (y - y_pred))
    db = (-2/n) * np.sum(y - y_pred)
    m = m - learning_rate * dm
    b = b - learning_rate * db
print("Slope (m):", m)
print("Intercept (b):", b)
y_pred = m * X + b
plt.scatter(X, y)
plt.plot(X, y_pred)
plt.xlabel("R&D Spend (Normalized)")
plt.ylabel("Profit")
plt.title("Gradient Descent on 50_Startups Dataset")
plt.show()

```

## Output:
<img width="623" height="470" alt="image" src="https://github.com/user-attachments/assets/ca4853ff-6210-4824-9889-d82fc34b4066" />



## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
