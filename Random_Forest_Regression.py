#Data Preprocessing
#Importing the libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing Dataset
dataset = pd.read_csv("Position_Salaries.csv")

#Independent Variable Matrix/ Vector
X = dataset.iloc[:,1:2].values

#Making Dependent Variable Matrix/ Vector
y= dataset.iloc[:, 2].values

#Fitting the model to the dataset
#create regressor
from sklearn.ensemble import RandomForestRegressor
regressor =RandomForestRegressor(random_state=0, n_estimators=500)
regressor.fit(X,y)

#Predicting Single Value/ new result with regression
y_pred = regressor.predict(np.array(6.5).reshape(1,-1))


#Visualising the Random Forest Regression Results with more division
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape(len(X_grid),1)
plt.scatter(X,y, color='red')
plt.plot(X_grid, regressor.predict(X_grid), color='purple')
plt.title("Salary vs Levels (Random Forest Regression)")
plt.xlabel("Levels")
plt.ylabel("Salary")
plt.show()