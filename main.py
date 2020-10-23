import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets,linear_model
from sklearn.metrics import mean_squared_error

# load the data of diabetes from datasets
diabetes = datasets.load_diabetes()

# load the data of index 2 of diabetes to diabetes_X
diabetes_X = diabetes.data[:,np.newaxis,2]

# print(diabetes)
# print(diabetes.keys())
# print(diabetes_X)


# collect the last 30 data as training data from diabetes_X
diabetes_X_train = diabetes_X[:-30]

# collect the first 30 data as testing data from diabetes_X
diabetes_X_test = diabetes_X[-30:]

# datasets of required or actual output
diabetes_Y_train = diabetes.target[:-30]
diabetes_Y_test = diabetes.target[-30:]

# create the model for linear regression
model = linear_model.LinearRegression()

# put the values in model by fit() method
model.fit(diabetes_X_train, diabetes_Y_train)

# check the predicted values that come from the regression by using the values of diabetes_X_test
diabetes_Y_predicted = model.predict(diabetes_X_test)

# printing mean squared error
print("Mean squared error is", mean_squared_error(diabetes_Y_test, diabetes_Y_predicted))

# printing weights and intercept
print("weights are:", model.coef_)
print("intercept:", model.intercept_)

# Scatter plots are used to plot data points on horizontal and vertical axis in the attempt
# to show how much one variable is affected by another.
plt.scatter(diabetes_X_test, diabetes_Y_test)

plt.plot(diabetes_X_test, diabetes_Y_predicted)

plt.show()

