import fastf1
import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt

# fastf1.Cache.enable_cache("f1_cache")

# load the 2024 Monaco Practice 2 session data
session = fastf1.get_session(2025, 9, "FP2")
session.load()

# Get accurate, non-pit laps
laps = session.laps.pick_accurate()
laps = laps[laps['PitInTime'].isna() & laps['PitOutTime'].isna()]

# Drop rows with missing compound info
laps = laps.dropna(subset=['Compound'])

# Choose the tire compound
target_compound = 'MEDIUM'

# Filter for this compound
laps = laps[laps['Compound'] == target_compound]

# Convert laptimes to seconds
laps['LapTime (s)'] = laps['LapTime'].dt.total_seconds()

# Get the drivers that used this compound
drivers = [
    d for d in laps['Driver'].unique()
    if len(laps[laps['Driver'] == d]) > 3
]

# Compute average pace
avg_pace = laps[laps['Driver'].isin(drivers)].groupby('Driver')['LapTime (s)'].mean()

# Normalize lap times by track length for comparison (Jeddah 6.174 km)
# track_length_km = 3.337
# avg_race_pace_normalized = avg_pace / track_length_km

# Create dataframe with the average race pace and normalized race pace
race_pace_df = pd.DataFrame(
    {"Average Race Pace (s)": avg_pace,
    #  "Normalized Race Pace (s/Km)": avg_race_pace_normalized
    }
).sort_values("Average Race Pace (s)")

# print(race_pace_df)

# Print result in racepace.py format
print("# clean air race pace from racepace.py")
print("clean_air_race_pace = {")

for driver, pace in race_pace_df["Average Race Pace (s)"].items():
    print(f'    "{driver}": {pace:.6f},')

print("}")

print(race_pace_df)
