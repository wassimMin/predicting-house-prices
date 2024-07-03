import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score
import tkinter as tk
from tkinter import messagebox


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

print(data.columns)

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
def predict_price():
    try:
        area = float(entry_area.get())
        bedrooms = int(entry_bedrooms.get())
        bathrooms = int(entry_bathrooms.get())
        stories = int(entry_stories.get())
        mainroad = int(entry_mainroad.get())
        guestroom = int(entry_guestroom.get())
        basement = int(entry_basement.get())
        hotwaterheating = int(entry_hotwaterheating.get())
        airconditioning = int(entry_airconditioning.get())
        parking = int(entry_parking.get())
        prefarea = int(entry_prefarea.get())
        furnishingstatus_semi_furnished = int(entry_furnishingstatus_semi_furnished.get())
        furnishingstatus_unfurnished = int(entry_furnishingstatus_unfurnished.get())
        
        input_data = [[area, bedrooms, bathrooms, stories, mainroad, guestroom, basement, hotwaterheating, airconditioning, parking, prefarea, furnishingstatus_semi_furnished, furnishingstatus_unfurnished]]
        input_data_scaled = scaler.transform(input_data)

        prediction = model.predict(input_data_scaled)
        messagebox.showinfo("Predicted Price", f"The predicted price is: {prediction[0]:}")
    except ValueError as e:
        messagebox.showerror("Input error", f"Invalid input: {e}")

# Create the main window
root = tk.Tk()
root.title("House Price Prediction")

# Create and place the input fields
labels = ['Area', 'Bedrooms', 'Bathrooms', 'Stories', 'Mainroad (1/0)', 'Guestroom (1/0)', 'Basement (1/0)',
          'Hot Water Heating (1/0)', 'Air Conditioning (1/0)', 'Parking', 'Prefarea (1/0)',
          'Furnishingstatus Semi-Furnished (1/0)', 'Furnishingstatus Unfurnished (1/0)']

entries = []
for label in labels:
    frame = tk.Frame(root)
    frame.pack(pady=5)
    lbl = tk.Label(frame, text=label)
    lbl.pack(side=tk.LEFT)
    entry = tk.Entry(frame)
    entry.pack(side=tk.RIGHT)
    entries.append(entry)

(entry_area, entry_bedrooms, entry_bathrooms, entry_stories, entry_mainroad, entry_guestroom, entry_basement, entry_hotwaterheating, 
entry_airconditioning, entry_parking, entry_prefarea, entry_furnishingstatus_semi_furnished, entry_furnishingstatus_unfurnished) = entries

# Create and place the predict button
btn_predict = tk.Button(root, text="Predict Price", command=predict_price)
btn_predict.pack(pady=20)

# Start the main event loop
root.mainloop()