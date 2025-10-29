import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler




 
 
imputer = SimpleImputer(missing_values=np.nan, strategy= 'mean')


dataset = pd.read_csv('Data.csv')
#matrix of feature
x= dataset.iloc[:,:-1].values
#dependent variable vector
y = dataset.iloc[:,-1].values

imputer.fit(x[:, 1:3])
x[:, 1:3] = imputer.transform(x[:, 1:3])

print(x)
print("Nan's are cleaned.")
print()

ct= ColumnTransformer(transformers= [('encoder',OneHotEncoder(),[0])], remainder='passthrough')
x = np.array(ct.fit_transform(x))

print(x)
print("Countries encoded.")
print()

le = LabelEncoder()
y = le.fit_transform(y)

print(y)
print("Results are  now binary based")
print()

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=1)
print('x_train')
print(x_train)
print('x_test')
print(x_test)
print('y_train')
print(y_train)
print('y_test')
print(y_test)
 
print('Dataset splitted.')
print()

sc = StandardScaler()
x_train[:,3:] = sc.fit_transform(x_train[:,3:])
x_test[:,3:] = sc.transform(x_test[:,3:])

print(x_train)
print(x_test)
print('Independent variables scaled.')
print()






