import pandas as pd 
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


dataset = pd.read_csv('winequality-red.csv',delimiter=';')


X = dataset.iloc[:, :-1].values
y = dataset.iloc[:,-1].values


X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state =42)



sc =StandardScaler()

X_train = sc.fit_transform(X_train)



X_test = sc.transform(X_test)


print(X_train)
print(X_test)