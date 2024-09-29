import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('/content/drive/MyDrive/CMaps/test_FD001.txt', delimiter=' ', header=None)

# Add column names
columns = ['unit_number', 'time_in_cycles'] + [f'sensor_{i}' for i in range(1, 26)] + ['RUL']
data.columns = columns

# Create RUL (Remaining Useful Life) based on max cycles
rul = data.groupby('unit_number')['time_in_cycles'].max().reset_index()
rul.columns = ['unit_number', 'max_cycles']
data = data.merge(rul, on=['unit_number'], how='left')
data['RUL'] = data['max_cycles'] - data['time_in_cycles']
data.drop('max_cycles', axis=1, inplace=True)

# Feature Engineering - Example: Rolling means (additional wear and tear indicators)
data['sensor_1_rolling_mean'] = data.groupby('unit_number')['sensor_1'].rolling(window=5).mean().reset_index(drop=True)
data['sensor_2_rolling_mean'] = data.groupby('unit_number')['sensor_2'].rolling(window=5).mean().reset_index(drop=True)
data.fillna(method='bfill', inplace=True)  # Handle NaN values created by rolling

# Drop non-feature columns
X = data.drop(['unit_number', 'time_in_cycles', 'RUL'], axis=1)
y = data['RUL']

# Handle missing values (if any) using SimpleImputer
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# Split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)

# Train a Random Forest model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Predict on the test set
y_pred = rf_model.predict(X_test)

# Evaluation Metrics
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Squared Error (MSE): {mse}")
print(f"Mean Absolute Error (MAE): {mae}")
# Nonconformance P-Chart (Monitor nonconforming predictions)
absolute_difference = abs(y_pred - y_test)
threshold = 10  # Define a threshold for nonconformance
nonconformance = (absolute_difference > threshold).astype(int)
p_values = nonconformance.mean()

plt.figure(figsize=(8, 6))
plt.bar(['Nonconformance'], [p_values], color='blue', alpha=0.7)
plt.axhline(y=p_values, color='red', linestyle='--', label='Overall Proportion')
plt.title('P-chart for Nonconformance')
plt.xlabel('Groups')
plt.ylabel('Proportion of Nonconforming Items')
plt.legend()
plt.ylim(0, 1)
plt.grid(True)
plt.show()
