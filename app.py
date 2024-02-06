import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import joblib
from statsmodels.tsa.statespace.sarimax import SARIMAX


loaded_model = joblib.load('trained_model.joblib')

import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

st.title('Price Prediction Dashboard')

default_main_city = ''
default_numerical_values = {feature: float(X[feature].mean()) for feature in numerical_features}

city_list = [''] + sorted(list(X['main_city'].unique())) 
main_city_value = st.selectbox('Select main_city', city_list, key='main_city')
if not main_city_value:
    st.warning('Please Select the City')

# Store the selected city for later use
selected_city = main_city_value

st.header('Input Features')

for feature in numerical_features:
    if feature != 'main_city':
        value = st.slider(f'Select Nearest Distance for a {feature}', float(X[feature].min()), float(X[feature].max()), 0.01)
        X[feature] = value

if st.button('Predict'):
    if not selected_city:
        st.warning('Please Select the City')
    else:
        # Use the stored city value for prediction
        X['main_city'] = selected_city
        prediction = loaded_model.predict(X)
        rounded_prediction = round(prediction[0] / 1000) * 1000  # Round to the nearest 1000
        st.write(f'Ideal Value for your Land : {rounded_prediction}')
        st.write(f'You can sell your Land in the range of : {rounded_prediction - rounded_prediction*0.5} to {rounded_prediction + rounded_prediction*0.5}')

time_series_df = pd.read_csv('13_for_modeling.csv')

st.set_option('deprecation.showPyplotGlobalUse', False)

if st.button('See your land price in next 3 years'):
    st.subheader('Land Price Forecast for the Next 3 Years')

    # Use the stored city value for time series analysis
    user_city = selected_city

    city_df = time_series_df[time_series_df['main_city'] == user_city]
    city_df['posted_date'] = pd.to_datetime(city_df['posted_date'])
    city_df.set_index('posted_date', inplace=True)

    quarterly_avg = city_df['price_land_pp'].resample('Q').mean()

    train_size = int(len(quarterly_avg) * 0.8)
    train, test = quarterly_avg[:train_size], quarterly_avg[train_size:]

    order = (1, 1, 1)
    seasonal_order = (1, 1, 1, 8)
    model = SARIMAX(train, order=order, seasonal_order=seasonal_order)
    model_fit = model.fit(method='powell')

    forecast_steps = 16  # Forecast for the next 3 years (12 quarters)
    forecast = model_fit.get_forecast(steps=forecast_steps)

    forecast_index = pd.date_range(start=quarterly_avg.index[-1] + pd.DateOffset(months=3), periods=forecast_steps, freq='Q')

    smoothed_forecast = forecast.predicted_mean.rolling(window=5, min_periods=1).mean()
    smoothed_forecast = smoothed_forecast.shift(-2)

    plt.figure(figsize=(12, 6))
    plt.plot(forecast_index, smoothed_forecast, color='red', label='Forecasted Land Price per Perch', linestyle='--')

    plt.title(f"Land Price Forecast for {user_city}")
    plt.xlabel('Time')
    plt.ylabel('Average Land Price')
    plt.legend()
    plt.show()
    st.pyplot()


