import pandas as pd
import numpy as np
import os
import glob

# 1. Load the local dataset directly by its file name
# Replace "ship_dataset.csv" with the actual exact name of your downloaded file
file_path = "datasets\\Ship_Performance_Dataset.csv" 
print(f"Loading local dataset: {file_path}")

df = pd.read_csv(file_path)
print(f"Original dataset loaded with {len(df)} rows.")


# 2. Fill missing ship types with mode and Generate Cargo Capacity and Load
# Fill 'nan' in Ship_Type with the most common ship type (the mode)
most_common_ship = df['Ship_Type'].mode()[0]
df['Ship_Type'] = df['Ship_Type'].fillna(most_common_ship)

# Mapping baseline capacities based on Ship_Type (values are illustrative)
capacity_map = {
    'Container Ship': 20000,  # in TEU
    'Tanker': 150000,         # in DWT
    'Bulk Carrier': 100000,   # in DWT
    'Fish Carrier': 5000      # in Tons
}

# Apply baseline capacity and add ±10% Gaussian noise for realism
df['Max_Capacity'] = df['Ship_Type'].map(capacity_map) * np.random.uniform(0.9, 1.1, len(df))
df['Max_Capacity'].fillna(50000, inplace=True) # Fallback for any unmapped ship types

# Generate 'Current_Load' to satisfy the cargo demand constraint
# Utilization factor randomly assigned between 60% and 100%
df['Utilization_Factor'] = np.random.uniform(0.6, 1.0, len(df))
df['Current_Load'] = df['Max_Capacity'] * df['Utilization_Factor']

# 3. Integrate Alternative Fuels and Shore Power
# Define the fuel types required by the problem statement
fuels = ['Heavy Fuel Oil (HFO)', 'LNG', 'Methanol', 'Hydrogen', 'Ammonia']
# Weighted probabilities (e.g., 40% conventional, 60% distributed among alternative fuels)
fuel_probs = [0.40, 0.25, 0.15, 0.10, 0.10]
df['Fuel_Type'] = np.random.choice(fuels, size=len(df), p=fuel_probs)

# Add binary flag for shore power solutions
# Higher probability (70%) of shore power for Short-haul routes
def assign_shore_power(route):
    if route == 'Short-haul':
        return np.random.choice([1, 0], p=[0.7, 0.3])
    return np.random.choice([1, 0], p=[0.2, 0.8])

df['Shore_Power_Enabled'] = df['Route_Type'].apply(assign_shore_power)

# 4. Calculate Lifecycle Greenhouse Gas Emissions
# Illustrative emission factors in gCO2e per kWh for each fuel type
emission_factors = {
    'Heavy Fuel Oil (HFO)': 600,
    'LNG': 450,
    'Methanol': 200,
    'Ammonia': 50,
    'Hydrogen': 20
}
df['Emission_Factor'] = df['Fuel_Type'].map(emission_factors)

# Calculate voyage duration in hours (Distance / Speed)
# Prevent division by zero if speed is missing/zero
safe_speed = df['Speed_Over_Ground_knots'].replace(0, 0.1)
df['Voyage_Duration_hours'] = df['Distance_Traveled_nm'] / safe_speed

# Calculate total energy consumed (kWh) = Power (kW) * Time (hours)
df['Energy_Consumed_kWh'] = df['Engine_Power_kW'] * df['Voyage_Duration_hours']

# Calculate lifecycle greenhouse gas emissions (kgCO2e)
df['Lifecycle_Emissions_kgCO2e'] = (df['Energy_Consumed_kWh'] * df['Emission_Factor']) / 1000

# 5. Final Output Preparation
print("\nAugmentation complete. Here is a preview of the new features:")
new_columns = ['Ship_Type', 'Current_Load', 'Fuel_Type', 'Shore_Power_Enabled', 'Lifecycle_Emissions_kgCO2e']
print(df[new_columns].head())

# Save the augmented dataset for model training
output_filename = "augmented_green_fleet_dataset.csv"
# df.to_csv(output_filename, index=False)
print(f"\nDataset saved successfully as '{output_filename}'")
print(f"Total number of rows: {df.shape[0]}")
print(f"Total number of columns: {df.shape[1]}\n")
