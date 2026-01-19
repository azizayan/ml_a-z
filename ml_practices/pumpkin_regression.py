import pandas as pd
import numpy as np
import seaborn as sns
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


score = lin_reg.score(X_test,y_test)
print('Model determination: ', score)



plt.scatter(X_test,y_test)
plt.plot(X_test,pred)
#plt.show()

#POLYNOMİAL REGRESSİON

pipeline = make_pipeline(PolynomialFeatures(2), LinearRegression())

pipeline.fit(X_train,y_train)


pred_polynommial = pipeline.predict(X_test)

mse = np.sqrt(mean_squared_error(y_test,pred))
print(f'Mean error: {mse:3.3} ({mse/np.mean(pred)*100:3.3}%)')


score = pipeline.score(X_test,y_test)
print('Model determination: ', score)


#LOGISTIC REGRESSION


dataset = pd.read_csv("ml_practices/US-pumpkins.csv")

new_columns_to_select = ['City Name','Package','Variety','Origin','Item Size', 'Color'] 
new_pumpkins =dataset.loc[:,new_columns_to_select]

new_pumpkins.dropna(inplace=True)
new_pumpkins.info()
palette = {
    'ORANGE': 'orange',
    'WHITE' : 'wheat',
}
sns.catplot(
data=new_pumpkins, y="Variety", hue="Color", kind="count",
palette=palette, 
)
#plt.show()

from sklearn.preprocessing import OrdinalEncoder

item_size_categories = [['sml', 'med', 'med-lge', 'lge', 'xlge', 'jbo', 'exjbo']]
ordinal_features = ['Item Size']
ordinal_encoder = OrdinalEncoder(categories=item_size_categories)

from sklearn.preprocessing import OneHotEncoder

categorical_features = ['City Name', 'Package', 'Variety', 'Origin']
categorical_encoder = OneHotEncoder(sparse_output=False)


from sklearn.compose import ColumnTransformer

ct = ColumnTransformer(transformers=[('ord',ordinal_encoder,ordinal_features),('cat',categorical_encoder,categorical_features)])

ct.set_output(transform='pandas')
encoded_features = ct.fit_transform(new_pumpkins)

from sklearn.preprocessing import LabelEncoder

label_encoder =LabelEncoder()
encoded_label = label_encoder.fit_transform(new_pumpkins['Color'])
 

encoded_pumpkins = encoded_features.assign(Color=encoded_label)

X = encoded_pumpkins[encoded_pumpkins.columns.difference(['Color'])]

y = encoded_pumpkins['Color']

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size= 0.2, random_state= 0)

from sklearn.metrics import f1_score, classification_report 
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)

print(classification_report(y_test, predictions))
print('Predicted labels: ', predictions)
print('F1-score: ', f1_score(y_test, predictions))