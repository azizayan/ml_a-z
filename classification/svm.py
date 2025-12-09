import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('Social_Network_Ads.csv')

X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split( X,y ,test_size= 0.25, random_state= 0)

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn import svm

classifier = svm.SVC(kernel = 'linear',random_state= 0)
classifier.fit(X_train,y_train)

print(classifier.support_vectors_)

y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix,accuracy_score

cm =  confusion_matrix(y_test,y_pred)
print(cm)
print(accuracy_score(y_test, y_pred))



