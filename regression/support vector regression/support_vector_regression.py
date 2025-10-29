import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder




dataset = pd.read_csv('Position_Salaries.csv')

X= dataset.iloc[:, 1:-1].values
y = dataset.iloc[:,-1].values

sc_X =StandardScaler()
X = sc_X.fit_transform(X)



#reshape y as scaler expects 2d arrau
y = y.reshape(len(y),1)
sc_y = StandardScaler()
y = sc_y.fit_transform(y)



print(X)
print(" ")
print(y)



from sklearn.svm import SVR

regressor = SVR(kernel = 'rbf')
regressor.fit(X, y)


#imverse transforming
y_pred = sc_y.inverse_transform(regressor.predict(sc_X.transform([[6.5]])).reshape(-1 ,1))
print(y_pred)




plt.scatter(sc_y.inverse_transform(X), sc_y.inverse_transform(y), color='red')
plt.plot(sc_y.inverse_transform(X), sc_y.inverse_transform(regressor.predict(X).reshape(-1, 1)), color='blue')
plt.title('Salary vs Experience (SVR)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()





