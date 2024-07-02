import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score


data = pd.read_csv('dataset/Housing.csv')

# print(data.describe())
# Converting yes/no to 1/0

columns_to_map = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']

for column in columns_to_map:
    data[column] = data[column].str.strip().str.lower()

data[columns_to_map] = data[columns_to_map].fillna('no')

mapping = {'yes': 1, 'no': 0}

for column in columns_to_map:
    data[column] = data[column].map(mapping)

# print(data.isna().sum())

# Below Ligne yzid y9sm colums t3 data l 0 ou 1 format bah tsahl predict 
data = pd.get_dummies(data,columns=['furnishingstatus'],drop_first=True)

# hna definina x w y n7wso n predicto price 3la 7ssab others features
x = data.drop('price', axis=1)
y= data['price']


scaler = StandardScaler()
X_scaled = scaler.fit_transform(x)

# print("Original Data:\n", x)
# print("\nScaled Data:\n", X_scaled)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train the Model
model = LinearRegression()
model.fit(X_train, y_train)

# Make Predictions
y_pred = model.predict(X_test)

# Evaluate the Model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'\nMean Squared Error: {mse}')
print(f'R^2 Score: {r2}')