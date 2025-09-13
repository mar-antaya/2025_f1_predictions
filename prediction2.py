import fastf1
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

# Enable FastF1 caching
fastf1.Cache.enable_cache("f1_cache")

# Load 2024 Chinese GP race session
session_2024 = fastf1.get_session(2024, "China", "R")
session_2024.load()

# Extract lap and sector times
laps_2024 = session_2024.laps[["Driver", "LapTime", "Sector1Time", "Sector2Time", "Sector3Time"]].copy()
laps_2024.dropna(inplace=True)

# Convert times to seconds
for col in ["LapTime", "Sector1Time", "Sector2Time", "Sector3Time"]:
    laps_2024[f"{col} (s)"] = laps_2024[col].dt.total_seconds()

# Group by driver to get average sector times per driver
sector_times_2024 = laps_2024.groupby("Driver")[["Sector1Time (s)", "Sector2Time (s)", "Sector3Time (s)"]].mean().reset_index()

# 2025 Qualifying Data Chinese GP
qualifying_2025 = pd.DataFrame({
    "Driver": ["Oscar Piastri", "George Russell", "Lando Norris", "Max Verstappen", "Lewis Hamilton",
               "Charles Leclerc", "Isack Hadjar", "Andrea Kimi Antonelli", "Yuki Tsunoda", "Alexander Albon",
               "Esteban Ocon", "Nico Hülkenberg", "Fernando Alonso", "Lance Stroll", "Carlos Sainz Jr.",
               "Pierre Gasly", "Oliver Bearman", "Jack Doohan", "Gabriel Bortoleto", "Liam Lawson"],
    "QualifyingTime (s)": [90.641, 90.723, 90.793, 90.817, 90.927,
                           91.021, 91.079, 91.103, 91.638, 91.706,
                           91.625, 91.632, 91.688, 91.773, 91.840,
                           91.992, 92.018, 92.092, 92.141, 92.174]
})

# Map full names to FastF1 3-letter codes
driver_mapping = {
    "Oscar Piastri": "PIA", "George Russell": "RUS", "Lando Norris": "NOR", "Max Verstappen": "VER",
    "Lewis Hamilton": "HAM", "Charles Leclerc": "LEC", "Isack Hadjar": "HAD", "Andrea Kimi Antonelli": "ANT",
    "Yuki Tsunoda": "TSU", "Alexander Albon": "ALB", "Esteban Ocon": "OCO", "Nico Hülkenberg": "HUL",
    "Fernando Alonso": "ALO", "Lance Stroll": "STR", "Carlos Sainz Jr.": "SAI", "Pierre Gasly": "GAS",
    "Oliver Bearman": "BEA", "Jack Doohan": "DOO", "Gabriel Bortoleto": "BOR", "Liam Lawson": "LAW"
}

qualifying_2025["DriverCode"] = qualifying_2025["Driver"].map(driver_mapping)

# Merge qualifying data with sector times
merged_data = qualifying_2025.merge(sector_times_2024, left_on="DriverCode", right_on="Driver", how="left")

# Define feature set (Qualifying + Sector Times)
X = merged_data[["QualifyingTime (s)", "Sector1Time (s)", "Sector2Time (s)", "Sector3Time (s)"]].fillna(0)
y = laps_2024.groupby("Driver")["LapTime (s)"].mean().reset_index()["LapTime (s)"]

# Train Gradient Boosting Model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=38)
model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, random_state=38)
model.fit(X_train, y_train)

# Predict race times using 2025 qualifying and sector data
predicted_race_times = model.predict(X)
qualifying_2025["PredictedRaceTime (s)"] = predicted_race_times

# Rank drivers by predicted race time
qualifying_2025 = qualifying_2025.sort_values(by="PredictedRaceTime (s)")

# Print final predictions
print("\n🏁 Predicted 2025 Chinese GP Winner with New Drivers and Sector Times 🏁\n")
print(qualifying_2025[["Driver", "PredictedRaceTime (s)"]])

# Evaluate Model
y_pred = model.predict(X_test)
print(f"\n🔍 Model Error (MAE): {mean_absolute_error(y_test, y_pred):.2f} seconds")


def print_baku_2025_predictions():
    # Get MAE from model evaluation
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    
    # Get the sorted drivers (already sorted in the main code)
    sorted_drivers = qualifying_2025.copy()
    
    # Top 3 for Qualifying and Race
    quali_top3 = sorted_drivers.head(3).reset_index(drop=True)
    race_top3 = sorted_drivers.head(3).reset_index(drop=True)
    
    # Podium emojis
    podium = ["🥇", "🥈", "🥉"]
    
    print("\n🏁 2025 Azerbaijan Grand Prix (Baku) - Official F1 Prediction Sheet 🏁")
    print(f"Model MAE (uncertainty): ±{mae:.2f} seconds")
    print("═════════════════════════════════════════════════════════════════════")
    print("🔒 Qualifying Top 3")
    print("─────────────────────────────────────────────────────────────────────")
    
    # Get P1 time for reference
    p1_time = quali_top3.iloc[0]["PredictedRaceTime (s)"]
    
    for i, (_, row) in enumerate(quali_top3.iterrows()):
        name = row["Driver"]
        lap_time = row["PredictedRaceTime (s)"]
        
        # Calculate gaps
        gap_to_p1 = lap_time - p1_time
        
        # Determine position stability
        if i > 0:
            prev_time = quali_top3.iloc[i-1]["PredictedRaceTime (s)"]
            gap_to_prev = lap_time - prev_time
            stability = "⚠️ High uncertainty — may flip" if abs(gap_to_prev) <= mae else "✅ Relatively stable"
        else:
            gap_to_prev = 0
            stability = "✅ Relatively stable"
        
        print(f"{podium[i]} {name}")
        print(f"    Predicted Lap: {lap_time:.3f}s")
        if i > 0:
            print(f"    Gap to P1: +{gap_to_p1:.3f}s")
            print(f"    Gap to P{i}: +{gap_to_prev:.3f}s")
        print(f"    Position: {stability}")
        
        # One short reasoning line based on numbers only
        if i == 0:
            print("    • Fastest predicted lap time suits Baku's high-speed layout, gives clear advantage.")
        elif i == 1:
            print(f"    • {gap_to_p1:.3f}s gap to P1 suggests competitive pace on Baku's long straights.")
        elif i == 2:
            print(f"    • {gap_to_p1:.3f}s off pole pace indicates strong but not dominant performance.")
            
    print("═════════════════════════════════════════════════════════════════════")
    print("🏆 Race Result Top 3")
    print("─────────────────────────────────────────────────────────────────────")
    
    # Get P1 time for reference
    p1_time = race_top3.iloc[0]["PredictedRaceTime (s)"]
    
    for i, (_, row) in enumerate(race_top3.iterrows()):
        name = row["Driver"]
        lap_time = row["PredictedRaceTime (s)"]
        
        # Calculate gaps
        gap_to_p1 = lap_time - p1_time
        
        # Determine position stability
        if i > 0:
            prev_time = race_top3.iloc[i-1]["PredictedRaceTime (s)"]
            gap_to_prev = lap_time - prev_time
            stability = "⚠️ High uncertainty — may flip" if abs(gap_to_prev) <= mae else "✅ Relatively stable"
        else:
            gap_to_prev = 0
            stability = "✅ Relatively stable"
        
        print(f"{podium[i]} {name}")
        print(f"    Predicted Lap: {lap_time:.3f}s")
        if i > 0:
            print(f"    Gap to P1: +{gap_to_p1:.3f}s")
            print(f"    Gap to P{i}: +{gap_to_prev:.3f}s")
        print(f"    Position: {stability}")
        
        # One short reasoning line based on numbers only
        if i == 0:
            print("    • Race pace likely consistent with qualifying performance, predicting maintained advantage.")
        elif i == 1:
            print(f"    • {gap_to_p1:.3f}s gap over race distance reflects consistent pace in challenging Baku conditions.")
        elif i == 2:
            print(f"    • Predicted {gap_to_p1:.3f}s behind leader suggests strong but manageable deficit on race day.")
            
    print("═════════════════════════════════════════════════════════════════════\n")


# Print Baku 2025 predictions
print_baku_2025_predictions()
