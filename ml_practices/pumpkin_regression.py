import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime


dataset = pd.read_csv("ml_practices/US-pumpkins.csv")
dataset = dataset[dataset['Package'].str.contains('bushel',case=True, regex=True)]
print(dataset.head())
print(dataset.isnull().sum())
day_of_year = pd.to_datetime(dataset['Date']).apply(lambda dt: (dt-datetime(dt.year,1,1)).days)

columns_to_select = ['Package', 'Low Price','High Price', 'Date','Origin','Variety']

dataset = dataset.loc[:, columns_to_select]

avg_price = (dataset['Low Price'] + dataset['High Price']) / 2
month = pd.DatetimeIndex(dataset['Date']).month

new_dataset = pd.DataFrame({'Month':month, 'Package':dataset['Package'],'Origin':dataset['Origin'], 'Price': avg_price,'Day':day_of_year,'Variety':dataset['Variety'] })


print(new_dataset)


from sklearn.model_selection import train_test_split

X = new_dataset[['Package', 'Month', 'Origin']]
y = new_dataset['Price']

X_train, X_test,  y_train, y_test = train_test_split(X,y, test_size=0.2,random_state=0)

categorical_features = ['Origin','Package']

numerical_features = [ 'Month']

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline



pie_pumpkins = new_dataset[new_dataset['Variety']=='PIE TYPE']




pie_pumpkins.dropna(inplace=True)
pie_pumpkins.info()


X = pie_pumpkins['Day'].to_numpy().reshape(-1,1)
y = pie_pumpkins['Price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


lin_reg = LinearRegression()
lin_reg.fit(X_train,y_train)

pred = lin_reg.predict(X_test)

mse = np.sqrt(mean_squared_error(y_test,pred))
print(f'Mean error: {mse:3.3} ({mse/np.mean(pred)*100:3.3}%)')


score = lin_reg.score(X_train,y_train)
print('Model determination: ', score)

plt.scatter(X_test,y_test)
plt.plot(X_test,pred)
plt.show()

#POLYNOMİAL REGRESSİON

pipeline = make_pipeline(PolynomialFeatures(2), LinearRegression())

pipeline.fit(X_train,y_train)


pred_polynommial = pipeline.predict(X_test)

mse = np.sqrt(mean_squared_error(y_test,pred))
print(f'Mean error: {mse:3.3} ({mse/np.mean(pred)*100:3.3}%)')


score = pipeline.score(X_train,y_train)
print('Model determination: ', score)

plt.scatter(X_test,y_test)
plt.plot(X_test,pred_polynommial)
plt.show()







