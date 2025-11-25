import fastf1
import pandas as pd
import os

# Make sure cache folder exists
os.makedirs("f1_cache", exist_ok=True)
fastf1.Cache.enable_cache("f1_cache")

# Load Las Vegas 2024 Race
session = fastf1.get_session(2024, 22, "R")
session.load()

laps = session.laps.copy()

# --- CLEAN AIR FILTER ---
# Use only reliable laps:
#   - Accurate timing
#   - Lap time between 40s and 120s (filters in/out/pit laps)
#   - Only green flag laps (TrackStatus == "1")
clean_air = laps[
    (laps["IsAccurate"] == True) &
    (laps["LapTime"].dt.total_seconds() > 40) &
    (laps["LapTime"].dt.total_seconds() < 120) &
    (laps["TrackStatus"].isin(["1"]))
]

clean_air["LapTime (s)"] = clean_air["LapTime"].dt.total_seconds()

# Compute clean-air average pace
pace = clean_air.groupby("Driver")["LapTime (s)"].mean().sort_values()

# Convert to dictionary
pace_dict = {drv: float(f"{time:.6f}") for drv, time in pace.items()}

# Print formatted version (compact)
formatted = ", ".join([f'"{drv}": {time:.6f}' for drv, time in pace_dict.items()])
print("\n📌 Clean Air Race Pace (Dictionary Format):\n")
print(formatted)

# Print nicely, 5 per line
print("\n📌 Clean Air Race Pace (Pretty Format):\n")
items = [f'"{drv}": {time:.6f}' for drv, time in pace_dict.items()]
for i in range(0, len(items), 5):
    print(", ".join(items[i:i+5]) + ",")

# Optionally write to file (uncomment this to save it)
# with open("clean_air_pace_output.txt", "w") as f:
#     f.write(formatted)
