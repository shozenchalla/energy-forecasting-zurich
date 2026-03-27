# ⚡ Zurich Electricity Consumption Forecasting

A machine learning project that forecasts electricity consumption in Zurich, Switzerland using XGBoost and time series cross-validation. The model predicts future energy demand based on historical consumption data and weather features.

![Python](https://img.shields.io/badge/Python-3.9+-blue?style=flat-square&logo=python)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-orange?style=flat-square)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-red?style=flat-square)
![Pandas](https://img.shields.io/badge/Pandas-2.0+-green?style=flat-square)

---

## 📌 Project Overview

This project uses 15-minute interval electricity consumption data from Zurich (2015–2022) to:

- Explore and visualize consumption patterns by hour, day, and month
- Build an XGBoost regression model with time-based and lag features
- Validate the model using proper time series cross-validation (no data leakage)
- Forecast electricity consumption one year into the future (Aug 2022 – Aug 2023)

---

## 📊 Results

| Metric | Value |
|--------|-------|
| Cross-Validation RMSE (avg) | **3,875.62** |
| Fold 1 RMSE | 3,150.55 |
| Fold 2 RMSE | 3,423.40 |
| Fold 3 RMSE | 4,409.57 |
| Fold 4 RMSE | 5,022.75 |
| Fold 5 RMSE | 3,371.82 |

> Average error is approximately **4–6%** of typical consumption values (~60,000–100,000 units), indicating strong predictive performance.

---

## 📁 Project Structure

```
zurich-energy-forecasting/
├── README.md
├── requirements.txt
├── energy_forecasting.ipynb       ← Main notebook (EDA + modeling + forecasting)
└── data/
    └── zurich_electricity_consumption.csv
```

---

## 🔍 Dataset

The dataset contains 15-minute interval readings from 2015 to 2022 with the following columns:

| Column | Description |
|--------|-------------|
| `Timestamp` | Date and time of reading |
| `Value_NE5` | Electricity consumption — network level 5 |
| `Value_NE7` | Electricity consumption — network level 7 |
| `T [°C]` | Temperature |
| `Hr [%Hr]` | Relative humidity |
| `RainDur [min]` | Rainfall duration |
| `StrGlo [W/m2]` | Global solar radiation |
| `WVs [m/s]` | Wind speed (scalar) |
| `p [hPa]` | Air pressure |

A `total_consumption` column was engineered as the sum of `Value_NE5` and `Value_NE7`.

---

## 🛠️ Methodology

### 1. Exploratory Data Analysis
- Visualized raw consumption over the full date range
- Identified and removed outliers (readings below 40,000 units)
- Analyzed consumption patterns by hour and month using boxplots

### 2. Feature Engineering
The following features were created from the timestamp index:

- **Time features**: hour, day of week, month, quarter, year, day of year, day of month, week of year
- **Lag features**: consumption from 1 year ago, 2 years ago, and 3 years ago — to capture yearly seasonality

### 3. Model Training
- **Algorithm**: XGBoost Regressor (`gbtree` booster)
- **Hyperparameters**: `n_estimators=2000`, `max_depth=3`, `learning_rate=0.01`
- **Validation**: 5-fold `TimeSeriesSplit` with a gap to prevent leakage

### 4. Forecasting
- Retrained the model on the full dataset
- Generated future timestamps from Aug 2022 to Aug 2023
- Applied the same feature engineering and lag mapping to future dates

---

## 🚀 How to Run

1. **Clone the repository**
```bash
git clone https://github.com/your-username/zurich-energy-forecasting.git
cd zurich-energy-forecasting
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Launch the notebook**
```bash
jupyter notebook energy_forecasting.ipynb
```

---

## 📦 Requirements

```
pandas
numpy
matplotlib
seaborn
xgboost
scikit-learn
jupyter
```

Or install all at once:
```bash
pip install -r requirements.txt
```

---

## 💡 Potential Improvements

- Incorporate weather features (temperature, humidity, solar radiation) already present in the dataset
- Add `is_weekend` and `is_holiday` binary features
- Experiment with LSTM or Prophet for comparison
- Hyperparameter tuning with Optuna or GridSearchCV

---

## 👤 Author

**Shozeb** — [GitHub Profile](https://github.com/your-username)

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
