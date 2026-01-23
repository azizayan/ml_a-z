import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime


dataset = pd.read_csv("ml_practices/US-pumpkins.csv")

dataset.info()

columns_to_select = ['City Name','Variety','Date','Origin','Item Size','Color']

dataset = dataset.loc[:, columns_to_select]
 
month = pd.DatetimeIndex(dataset['Date']).month

dataset = pd.DataFrame({'Month':month,'Origin':dataset['Origin'], 'City Name': dataset['City Name'],'Variety':dataset['Variety'],'Item Size':dataset['Item Size'],'Color':dataset['Color'] })




#City Name-> the region that pumpkin grows could has an effect on the color of it
#Variety->directly correlated to color of pumpkin
#Date-> weather conditions changes due to date, so date has an possible effect on the color of pumpkin
#Origin->directly correlated to color of pumpkin
#Item Size-> It may has an effect on the color of pumpkin

#Variety 5 missing, object
#Origin 3 missing, object
#Item Size 270 missing, object
#Color 610 missing, object. 
# Result column has so much missing so using simple imputer effects the results with a high tendency to most frequent color, so we dont

dataset = dataset.dropna(subset=['Color'])

from sklearn.model_selection import train_test_split

X = dataset[['City Name','Variety','Month','Origin','Item Size']]
y = dataset['Color']

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size= 0.2,random_state=2)


categorical_features = ['Origin','Variety','City Name','Item Size']
numerical_fetaures = ['Month']

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore'))
        ]), categorical_features),

        ('num', SimpleImputer(strategy='median'), numerical_fetaures)
    ]
)

from sklearn.linear_model import LogisticRegression

model = Pipeline(steps=[
    ('preprocess',preprocessor),
    ('train',LogisticRegression(max_iter=1000))
])


model.fit(X_train,y_train)


from sklearn.metrics import accuracy_score, classification_report

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
