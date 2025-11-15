import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


dataset = pd.read_csv('Data_Evaluation.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size= 0.2, random_state=0)

from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators= 10 ,random_state= 4)
regressor.fit(X_train,y_train)


y_pred = regressor.predict(X_test)

from sklearn.metrics import r2_score


r2 = r2_score(y_test, y_pred)
print('r2 score for this model is : ' ,r2)