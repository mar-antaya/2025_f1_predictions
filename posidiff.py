import fastf1
import pandas as pd
import os

# ---------------------------------------
# Ensure FastF1 cache is available
# ---------------------------------------
os.makedirs("f1_cache", exist_ok=True)
fastf1.Cache.enable_cache("f1_cache")

# ---------------------------------------
# Load Las Vegas 2024 Qualifying
# ---------------------------------------
session_q = fastf1.get_session(2024, 22, "Q")
session_q.load()

laps_q = session_q.laps.copy()

# Best qualifying lap per driver
quali_best = (
    laps_q[["Driver", "LapTime"]]
    .dropna()
    .groupby("Driver")["LapTime"]
    .min()
    .sort_values()
)

quali_positions = {drv: i + 1 for i, drv in enumerate(quali_best.index)}

# ---------------------------------------
# Load Las Vegas 2024 Race Results
# ---------------------------------------
session_r = fastf1.get_session(2024, 22, "R")
session_r.load()

results = session_r.results

# 🔍 DEBUG: print available columns
print("\nResults Columns:", results.columns.tolist(), "\n")

# Correct fields:
# DriverNumber → Abbreviation → Position
driver_codes = {
    row["DriverNumber"]: row["Abbreviation"]
    for _, row in results.iterrows()
}

finishing_positions = {
    row["DriverNumber"]: row["Position"]
    for _, row in results.iterrows()
}

race_positions_code = {
    driver_codes[num]: pos
    for num, pos in finishing_positions.items()
}

# ---------------------------------------
# Calculate position changes
# ---------------------------------------
position_change = {}
for driver, start_pos in quali_positions.items():
    finish_pos = race_positions_code.get(driver)
    if finish_pos is not None:
        position_change[driver] = float(f"{finish_pos - start_pos:.1f}")

# ---------------------------------------
# Print dictionary
# ---------------------------------------
print("========== LAS VEGAS 2024 POSITION CHANGE ==========\n")

formatted = ", ".join([f'"{drv}": {chg}' for drv, chg in position_change.items()])
print("📌 Compact Format:\n")
print(formatted)

print("\n📌 Pretty Format:\n")
items = [f'"{drv}": {chg}' for drv, chg in position_change.items()]
for i in range(0, len(items), 5):
    print(", ".join(items[i:i+5]) + ",")

average_position_change = position_change
__all__ = ["average_position_change"]
