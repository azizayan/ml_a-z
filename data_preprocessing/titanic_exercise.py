
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder


dataset = pd.read_csv('titanic.csv')





categorical_faetures=['Sex','Embarked','Pclass']



ct= ColumnTransformer(transformers=[('encoder',OneHotEncoder(),categorical_faetures)],remainder= 'passthrough')


X = ct.fit_transform(dataset)



X = np.array(X)


le = LabelEncoder()
y = le.fit_transform(dataset['Survived'])

print(X)
print(y)
