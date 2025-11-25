import fastf1
import pandas as pd
import os

 
# Ensure cache exists
 
os.makedirs("f1_cache", exist_ok=True)
fastf1.Cache.enable_cache("f1_cache")

 
# Load Las Vegas 2025 Qualifying

session_q = fastf1.get_session(2025, 22, "Q")  # Round 22 = Las Vegas GP
session_q.load()

laps_q = session_q.laps.copy()

 
# Extract best quali lap per driver
 
quali_times = (
    laps_q[["Driver", "LapTime"]]
    .dropna()
    .groupby("Driver")["LapTime"]
    .min()
    .dt.total_seconds()
)

# Convert to normal Python dict with 6 decimals
qualifying_times = {drv: float(f"{t:.6f}") for drv, t in quali_times.items()}

 
# PRINT FORMATTED RESULTS
 

print("\n========== LAS VEGAS 2025 QUALIFYING TIMES ==========\n")
print(quali_times)

# COMPACT one-line dictionary style:
formatted = ", ".join([f'"{drv}": {time:.6f}' for drv, time in qualifying_times.items()])

print("\n📌 Compact Dictionary Format:\n")
print(formatted)

# PRETTY multi-line version (5 per line)
print("\n📌 Pretty 5-Per-Line Format:\n")
items = [f'"{drv}": {time:.6f}' for drv, time in qualifying_times.items()]
for i in range(0, len(items), 5):
    print(", ".join(items[i:i+5]) + ",")

 
# Make it importable for other scripts
 
__all__ = ["qualifying_times"]
