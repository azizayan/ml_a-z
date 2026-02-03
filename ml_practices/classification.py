import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mlp
import numpy as np
import kagglehub 
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split



df = pd.read_csv('asian_indian_recipes.csv')

print(df.head())
df.info()


#df.cuisine.value_counts().plot.barh()
#plt.show()


thai_df = df[(df.cuisine== "thai")]
japanese_df = df[(df.cuisine== "japanese")]
chinese_df = df[(df.cuisine == "chinese")]
indian_df = df[(df.cuisine == "indian")]
korean_df = df[(df.cuisine == "korean")]


print(f'thai df: {thai_df.shape}')
print(f'japanese df: {japanese_df.shape}')
print(f'chinese df: {chinese_df.shape}')
print(f'indian df: {indian_df.shape}')
print(f'korean df: {korean_df.shape}')

def create_ingredient_df(df):
    ingredient_df = df.T.drop(['cuisine','Unnamed: 0']).sum(axis=1).to_frame('value')
    ingredient_df = ingredient_df[(ingredient_df.T != 0).any()]
    ingredient_df = ingredient_df.sort_values(by='value', ascending=False,
    inplace=False)
    return ingredient_df

feature_df = df.drop(['cuisine','Unnamed: 0','rice', 'garlic', 'ginger'], axis = 1)
labels_df = df.cuisine

X= feature_df
y = labels_df


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state= 0)

oversample = SMOTE()
transformed_feature_df, transformed_label_df = oversample.fit_resample(X_train, y_train)


print(f'new label count: {transformed_label_df.value_counts()}')
print(f'old label count: {df.cuisine.value_counts()}')


transformed_df = pd.concat([transformed_label_df,transformed_feature_df], axis=1, join='outer')

transformed_df.head()
transformed_df.info()
transformed_df.to_csv("cleaned_cuisines.csv")


cuisines_df = pd.read_csv('cleaned_cuisines.csv')
cuisines_df.info()
print(cuisines_df.head)

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix, classification_report,precision_recall_curve
from sklearn.svm import SVC

cuisines_label_df = cuisines_df['cuisine']
cuisines_label_df.head()

cuisines_feature_df = cuisines_df.drop(['Unnamed: 0','cuisine'],axis = 1)

X_train, X_test, y_train, y_test = train_test_split(cuisines_feature_df, cuisines_label_df, test_size=0.3)



lr = LogisticRegression(multi_class='ovr', solver='liblinear')
model = lr.fit(X_train, np.to_numpy(y_train))

accuracy = model.score(X_test, y_test)
print("Accuracy is {}".format(accuracy))