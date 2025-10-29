import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:,1:-1].values #already encoded as levels so 1:-1
y = dataset.iloc[:,-1].values



#linear regression
linear_regressor = LinearRegression()
linear_regressor.fit(X, y)


from sklearn.preprocessing import PolynomialFeatures
polynomial_regressor = PolynomialFeatures(degree = 4)


X_poly = polynomial_regressor.fit_transform(X)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly,y)



fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))


ax1.scatter(X, y, color='red')
ax1.plot(X, linear_regressor.predict(X), color='blue')
ax1.set_title('Salary vs Experience (Linear Regression)')
ax1.set_xlabel('Years of Experience')
ax1.set_ylabel('Salary')

# Polynomial regression plot
ax2.scatter(X, y, color='red')
ax2.plot(X, lin_reg_2.predict(X_poly), color='blue')
ax2.set_title('Salary vs Experience (Polynomial Regression)')
ax2.set_xlabel('Years of Experience')
ax2.set_ylabel('Salary')

plt.tight_layout()  
plt.show()



#PREDICTING RESULTS
y_pred = linear_regressor.predict([[8.5]])# predict fucntion takes array as an parameter
print(y_pred)

y_pred_2 = lin_reg_2.predict(polynomial_regressor.fit_transform([[6.5]]))
print(y_pred_2)
















