# 2025_f1_predictions

# 🏎️ F1 Predictions 2025 - Machine Learning Model

Welcome to the **F1 Predictions 2025** repository! This project uses **machine learning, FastF1 API data, and historical F1 race results** to predict race outcomes for the 2025 Formula 1 season.

## 🚀 Project Overview
This repository contains a **Gradient Boosting Machine Learning model** that predicts race results based on past performance, qualifying times, and other structured F1 data. The model leverages:
- FastF1 API for historical race data
- 2024 race results
- 2025 qualifying session results
- Over the course of the season we will be adding additional data to improve our model as well
- Feature engineering techniques to improve predictions

## 📊 Data Sources
- **FastF1 API**: Fetches lap times, race results, and telemetry data
- **2025 Qualifying Data**: Used for prediction
- **Historical F1 Results**: Processed from FastF1 for training the model

## 🏁 How It Works
1. **Data Collection**: The script pulls relevant F1 data using the FastF1 API.
2. **Preprocessing & Feature Engineering**: Converts lap times, normalizes driver names, and structures race data.
3. **Model Training**: A **Gradient Boosting Regressor** is trained using 2024 race results.
4. **Prediction**: The model predicts race times for 2025 and ranks drivers accordingly.
5. **Evaluation**: Model performance is measured using **Mean Absolute Error (MAE)**.

### Dependencies
- `fastf1`
- `numpy`
- `pandas`
- `scikit-learn`
- `matplotlib`

To download dependencies using following command:

```
pip install -r requirements.txt
```

## File Structure 
- For every race the end of the file will be numbered in correlation to the race on the calendar, ex. prediction1 - Australia, prediction2 - China, etc.

## 🔧 Usage
Run the prediction script:
```bash
python3 prediction1.py
```
Expected output:
```
🏁 Predicted 2025 Australian GP Winner 🏁
Driver: Charles Leclerc, Predicted Race Time: 82.67s
...
🔍 Model Error (MAE): 3.22 seconds
```

## 📈 Model Performance
The Mean Absolute Error (MAE) is used to evaluate how well the model predicts race times. Lower MAE values indicate more accurate predictions.

## 📌 Future Improvements
- Incorporate **weather conditions** as a feature
- Add **pit stop strategies** into the model
- Explore **deep learning** models for improved accuracy
- @mar_antaya on Instagram and TikTok will update with the latest predictions before every race of the 2025 F1 season

## 📜 License
This project is licensed under the MIT License.


🏎️ **Start predicting F1 races like a data scientist!** 🚀

