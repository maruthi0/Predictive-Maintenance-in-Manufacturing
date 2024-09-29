import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
df = pd.read_csv('/train_FD001.txt')
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Set random seed for reproducibility
np.random.seed(42)

# Number of data points
num_data_points = 1000

# Create synthetic sensor data
sensor_data = pd.DataFrame({
    'Timestamp': pd.date_range(start=datetime.now(), periods=num_data_points, freq='H'),
        'Temperature': np.random.normal(loc=25, scale=5, size=num_data_points),
            'Pressure': np.random.normal(loc=100, scale=10, size=num_data_points),
                'Vibration': np.random.normal(loc=0, scale=1, size=num_data_points),
                })

                # Create synthetic maintenance records
maintenance_records = pd.DataFrame({
     'Timestamp': pd.date_range(start=datetime.now(), periods=num_data_points//10, freq='D'),
         'Maintenance_Type': np.random.choice(['Routine', 'Emergency'], size=num_data_points//10),
         })

                        # Merge sensor data and maintenance records
df = pd.merge_asof(sensor_data, maintenance_records, on='Timestamp', direction='backward')

                        # Fill NaN values in 'Maintenance_Type' with 'No Maintenance'
df['Maintenance_Type'].fillna('No Maintenance', inplace=True)

                        # Create binary target variable indicating failure (1) or not (0)
df['Failure'] = np.where(df['Maintenance_Type'] == 'Emergency', 1, 0)

                        # Save the synthetic dataset to a CSV file
df.to_csv('data.csv', index=False)
# Explore the data
print(df.head())
print(df.info())
print(df.describe())

# Feature selection
features = ['Temperature', 'Pressure', 'Vibration']
X = df[features]
y = df['Failure']

# Handle missing values (if any)
X.fillna(X.mean(), inplace=True)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
