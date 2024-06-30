import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# Load the generated data
df = pd.read_csv("daily_water_quality_data.csv")

# Introduce a binary target variable (1: Potable, 0: Not Potable) for training purposes
np.random.seed(42)  # For reproducibility
df['Potable'] = np.random.choice([0, 1], size=len(df))

# Aggregate data by day to create daily averages
daily_data = df.groupby('Day').mean()

# Ensure the target variable is binary
daily_data['Potable'] = daily_data['Potable'].round().astype(int)

# Separate features and target
X = daily_data[['Temperature', 'Dissolved_Oxygen', 'pH', 'Dissolved_Solids', 'Suspended_Solids']]
y = daily_data['Potable']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define and train the MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(10, 10), max_iter=1000, random_state=42)
mlp.fit(X_train, y_train)

# Predict on the test set
y_pred = mlp.predict(X_test)

# Evaluate the classifier
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# Save the trained model (optional)
import joblib
model_path = "water_quality_classifier.pkl"
joblib.dump(mlp, model_path)
print(f"Model saved as {model_path}")

# Load the test day data
test_df = pd.read_csv("test_day_water_quality_data.csv")

# Aggregate the data for the single day to create averages
test_day_data = test_df.mean().to_frame().T

# Extract the actual label for the test day
expected_potable = test_df['Potable'].iloc[0]

# Ensure the same order of columns
test_day_data = test_day_data[['Temperature', 'Dissolved_Oxygen', 'pH', 'Dissolved_Solids', 'Suspended_Solids']]

# Predict the potability of the test day data
test_day_prediction = mlp.predict(test_day_data)
test_day_prediction == expected_potable

# Compare the predicted and expected results
print(f"Expected result: {'Potable' if expected_potable == 1 else 'Not Potable'}")
print(f"Predicted result: {'Potable' if test_day_prediction[0] == 1 else 'Not Potable'}")
