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

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression

preprocess = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(drop='first'), categorical_features),
        ('num', 'passthrough', numerical_features)
    ]
)

model = Pipeline(steps=[('preprocess',preprocess),('regressor',LinearRegression())])

model.fit(X_train,y_train)

y_pred = model.predict(X_test)

from sklearn.metrics import r2_score

r2 = r2_score(y_test, y_pred)
print("R²:", r2)


comparison = pd.DataFrame({
    "Actual Price": y_test.values,
    "Predicted Price": y_pred,
    "Error": y_pred - y_test.values
})

print(comparison.head(10))

from sklearn.metrics import mean_absolute_error

baseline_pred = np.full_like(y_test, y_train.mean())
print("Baseline MAE:",
      mean_absolute_error(y_test, baseline_pred))




plt.figure()
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()])
plt.show()

test_samples = pd.DataFrame({
    "Package": [
        "1/2 bushel cartons", "1/2 bushel cartons", "1 1/9 bushel cartons",
        "1/2 bushel cartons", "1 1/9 bushel cartons",

        "1 1/9 bushel cartons", "1 1/9 bushel cartons", "1/2 bushel cartons",
        "1 1/9 bushel cartons", "1/2 bushel cartons"
    ],
    "Month": [
        9, 10, 10, 11, 9,
        10, 11, 9, 10, 11
    ],
    "Origin": [
        "MICHIGAN", "OHIO", "ILLINOIS", "MICHIGAN", "INDIANA",
        "CALIFORNIA", "DELAWARE", "VIRGINIA", "CANADA", "CALIFORNIA"
    ]
})

predictions = model.predict(test_samples)

results = test_samples.copy()
results["Predicted Price"] = predictions

print(results)



ax = None
colors = ['red','blue','green','yellow']

for i, var in enumerate(new_dataset['Variety'].unique()):
    df = new_dataset[new_dataset['Variety'] == var]
    ax = df.plot.scatter('Day', 'Price', ax=ax,
                         c=colors[i % len(colors)],
                         label=var)

plt.show()

