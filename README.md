# Weather Prediction App

This is a weather prediction application built with Python and Streamlit. It uses machine learning to predict tomorrow's weather based on historical weather data.

## Features

- Upload CSV files containing weather data
- Real-time weather prediction
- Model performance metrics
- Feature importance visualization
- Interactive data preview

## Requirements

- Python 3.8 or higher
- Required packages (install using `pip install -r requirements.txt`):
  - streamlit
  - pandas
  - numpy
  - scikit-learn
  - joblib

## Installation

1. Clone this repository
2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```
2. Open your web browser and go to the URL shown in the terminal (usually http://localhost:8501)
3. Upload your weather data CSV file
4. View the predictions and model performance

## CSV File Format

Your CSV file should contain:
- Numerical weather data (e.g., temperature, humidity, pressure, etc.)
- The last column should be the target variable (e.g., temperature)
- Each row represents a day's weather data
- Column headers should be descriptive of the weather features

Example CSV format:
```
date,temperature,humidity,pressure,wind_speed
2024-01-01,25.5,65,1013,10
2024-01-02,26.0,70,1012,12
...
```

## Note

The application uses a Random Forest Regressor model for predictions. The model is trained on the uploaded data and makes predictions based on the most recent weather data. 