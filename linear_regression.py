import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.metrics import mean_squared_error

# Step 1: Load data from.csv file

dataframe=pd.read_csv("student_scores.csv")

# Step 2: Plot the data in a suitable form of graph

X=dataframe["Hours"].values.reshape(-1,1)
Y=dataframe["Scores"].values.reshape(-1,1)
(x_train,y_train)=(X[:20],Y[:20])
(x_test,y_test)=(X[20:],Y[20:])

# Step 4: Plot the Regression line

model=LinearRegression()
model.fit(x_train,y_train)

# Step 5: Make Predictions on Test Data

regression_line=model.predict(X)
plt.plot(x_train,y_train,'o')
plt.plot(x_test,y_test,'o')
plt.plot(X,regression_line)
predictions=model.predict(x_test)
plt.plot(x_test,predictions,'*')
plt.show()
# Step 6: Estimate Error

print("MSE: ",mean_squared_error(y_test,predictions))
