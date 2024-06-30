import pandas as pd
import numpy as np

# Generate random data for a single day (48 measurements)
data = {
    'Temperature': np.random.uniform(0, 35, 48),  # Temperature between 0 and 35 degrees Celsius
    'Dissolved_Oxygen': np.random.uniform(0, 14, 48),  # Dissolved oxygen between 0 and 14 mg/L
    'pH': np.random.uniform(6.5, 9, 48),  # pH between 6.5 and 9
    'Dissolved_Solids': np.random.uniform(0, 5000, 48),  # Dissolved solids between 0 and 5000 mg/L
    'Suspended_Solids': np.random.uniform(0, 1000, 48)  # Suspended solids between 0 and 1000 mg/L
}

# Create a DataFrame
df = pd.DataFrame(data)

# Introduce a binary target variable (1: Potable, 0: Not Potable) for testing purposes
# Here we set it manually for demonstration purposes, you can set it randomly if needed
potable = np.random.choice([0, 1])
df['Potable'] = potable

# Save the DataFrame to a CSV file
test_csv_path = "test_day_water_quality_data.csv"
df.to_csv(test_csv_path, index=False)

print(f"CSV file for test day has been saved as {test_csv_path}")
print(f"Expected result: {'Potable' if potable == 1 else 'Not Potable'}")
