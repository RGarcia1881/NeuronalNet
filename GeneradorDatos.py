import pandas as pd
import numpy as np

# Set the number of days
num_days = 365

# Generate random data for each day (48 measurements per day)
data = {
    'Day': np.repeat(np.arange(num_days), 48),
    'Measurement': np.tile(np.arange(48), num_days),
    'Temperature': np.random.uniform(0, 35, num_days * 48),  # Temperature between 0 and 35 degrees Celsius
    'Dissolved_Oxygen': np.random.uniform(0, 14, num_days * 48),  # Dissolved oxygen between 0 and 14 mg/L
    'pH': np.random.uniform(6.5, 9, num_days * 48),  # pH between 6.5 and 9
    'Dissolved_Solids': np.random.uniform(0, 5000, num_days * 48),  # Dissolved solids between 0 and 5000 mg/L
    'Suspended_Solids': np.random.uniform(0, 1000, num_days * 48)  # Suspended solids between 0 and 1000 mg/L
}

# Create a DataFrame
df = pd.DataFrame(data)

# Save the DataFrame to a CSV file
csv_path = "daily_water_quality_data.csv"
df.to_csv(csv_path, index=False)

print(f"CSV file has been saved as {csv_path}")
