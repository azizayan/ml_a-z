import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


dataset = pd.read_csv('/Users/azizemreayan/Desktop/my_ml_a-z_implementations/Cas.csv')


dataset.info()

for col in dataset.columns:
    print(col, dataset[col].unique()[:10])


dataset['Sex_of_Casualty'] = dataset['Sex_of_Casualty'].replace(-1, np.nan)

dataset['Pedestrian_Location'] = dataset['Pedestrian_Location'].replace(0, np.nan)

dataset['Pedestrian_Movement'] = dataset['Pedestrian_Movement'].replace(0, np.nan)

dataset['Bus_or_Coach_Passenger'] = dataset['Bus_or_Coach_Passenger'].replace(-1, np.nan)

dataset['Pedestrian_Road_Maintenance_Worker'] = dataset['Pedestrian_Road_Maintenance_Worker'].replace(-1, np.nan)

dataset['Casualty_Type'] = dataset['Casualty_Type'].replace(0, np.nan)

dataset['Casualty_Home_Area_Type'] = dataset['Casualty_Home_Area_Type'].replace(-1, np.nan)

dataset['Casualty_IMD_Decile'] = dataset['Casualty_IMD_Decile'].replace(-1, np.nan)

print(dataset.isna().sum())