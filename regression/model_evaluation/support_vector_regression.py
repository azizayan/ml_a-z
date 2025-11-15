import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


dataset = pd.read_csv('Data_Evaluation.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
y = y.reshape(len(y),1)


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size= 0.2, random_state=0)


sc_X =StandardScaler()
X_train = sc_X.fit_transform(X_train)


sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)


from sklearn.svm import SVR

regressor = SVR(kernel = 'rbf')
regressor.fit(X_train, y_train.ravel())


#imverse transforming
y_pred = sc_y.inverse_transform(regressor.predict(sc_X.transform(X_test)).reshape(-1 ,1))
#y_test_original = sc_y.inverse_transform(y_test.reshape(-1,1)), no need to this but why?





from sklearn.metrics import r2_score


r2 = r2_score(y_test, y_pred)
print('r2 score for this model is : ', r2)

