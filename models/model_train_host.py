import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib

import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('laptop_data.csv')
df_original = df.copy()

df.describe(include="all")


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder

# Preprocessing (assuming similar preprocessing as your original notebook)
df = df.drop(columns=['Unnamed: 0'])
df['Ram'] = df['Ram'].str.replace('GB', '')
df['Weight'] = df['Weight'].str.replace('kg', '')
df['Ram'] = df['Ram'].astype('int32')
df['Weight'] = df['Weight'].astype('float32')
df['Price'] = df['Price'].astype(int)

# Create a new column to store the Screen Resolution type
df['ScreenResolutionType'] = df['ScreenResolution'].str.extract(r'(IPS|Touchscreen)', expand=False).fillna('Other')

# Drop the original ScreenResolution column
df = df.drop(columns=['ScreenResolution']) # <---- Add this line to drop the original column

# One-hot encoding for categorical features
categorical_cols = ['Company', 'TypeName', 'OpSys', 'Cpu', 'Gpu', 'Memory', 'ScreenResolutionType'] # Include new column for encoding
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# ----> Extract Screen Resolution Feature <----
# This section is removed since it was causing the KeyError
# as 'ScreenResolution' column has been already dropped.
# The ScreenResolutionType column is already created and will be used in encoding

# Select features and target
X = df_encoded.drop('Price', axis=1)
y = df_encoded['Price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Train a RandomForestRegressor (or another suitable model)
model = RandomForestRegressor(n_estimators=100, random_state=42)  # Adjust parameters as needed
model.fit(X_train, y_train)

# Function to predict price based on user input
def predict_price(company, ram, memory, gpu, inch, os):
    # Create a DataFrame for the input features
    input_data = pd.DataFrame({'Ram': [ram], 'Weight': [0], 'Inches': [inch]}) # Initialize with some placeholders

    # Add one-hot encoded columns for categorical features.
    # Critical to keep columns consistent with training data
    for col in categorical_cols:
      if col == 'Company':
          temp_df = pd.get_dummies(pd.DataFrame({'Company':[company]}), columns=['Company'], drop_first=True)
      elif col == 'Memory':
          temp_df = pd.get_dummies(pd.DataFrame({'Memory':[memory]}), columns=['Memory'], drop_first=True)
      elif col == 'Gpu':
          temp_df = pd.get_dummies(pd.DataFrame({'Gpu':[gpu]}), columns=['Gpu'], drop_first=True)
      elif col == 'OpSys':
          temp_df = pd.get_dummies(pd.DataFrame({'OpSys':[os]}), columns=['OpSys'], drop_first=True)
      else: # For Cpu and TypeName
          temp_df = pd.DataFrame()
      input_data = pd.concat([input_data, temp_df], axis = 1)


    # Align columns with training data, fill missing columns with 0.
    input_data = input_data.reindex(columns=X_train.columns, fill_value=0)

    # Make the prediction
    predicted_price = model.predict(input_data)[0]
    return predicted_price


# Example usage:
company = 'HP'
ram = 8
memory = '128GB SSD'
gpu = 'Intel UHD Graphics'
inch = 15.6
os = 'Windows 10'
predicted_price = predict_price(company, ram, memory, gpu, inch, os)
print(f"Predicted price: {predicted_price:.2f}")

# prompt: check the accuracy and mse

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# Calculate accuracy (R-squared)
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_pred)
print(f"Accuracy (R-squared): {r2}")
import pickle
pickle.dump(model, open('model.pkl', 'wb'))