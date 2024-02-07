import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import joblib
from statsmodels.tsa.statespace.sarimax import SARIMAX
import plotly.express as px
from xgboost import XGBRegressor

to_filter_df = pd.read_csv('14_for_dashboard.csv')
columns_to_keep = ['main_city', 'price_land_pp', 'Hospital', 'National School', 'Railway Station', 'Bus Stop', 'Main Road']
filtered_df = to_filter_df[columns_to_keep].copy()

X = filtered_df.drop('price_land_pp', axis=1)
y = filtered_df['price_land_pp']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

numerical_features = X.select_dtypes(include=['float64', 'int64']).columns
categorical_features = ['main_city'] + list(X.select_dtypes(include=['object']).columns)

numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

model = XGBRegressor(n_estimators=50, learning_rate=0.1)
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                             ('model', model)])

pipeline.fit(X_train, y_train)


joblib.dump(pipeline, 'trained_model.joblib')


@st.cache_resource
def load_model():
    return joblib.load("trained_model.joblib")

@st.cache_data
def load_cities():
    f = pd.read_csv("City_list_File.csv")
    return f["Cities"].to_list()

loaded_model = load_model()

import matplotlib.pyplot as plt

st.title("Price Prediction Dashboard")

default_main_city = ""
# default_numerical_values = {feature: float(X[feature].mean()) for feature in numerical_features}

city_list = load_cities()
main_city_value = st.selectbox("Select main_city", city_list, key="main_city")
if not main_city_value:
    st.warning("Please Select the City")

# Store the selected city for later use
selected_city = main_city_value
Y = {}

st.header("Input Features")

# for feature in numerical_features:
#     if feature != 'main_city':
#        value = st.slider(f'Select Nearest Distance for a {feature}', float(X[feature].min()), float(X[feature].max()), 0.01)
#        Y[feature] = value

Y["main_city"] = [selected_city]
Y["Hospital"] = [
    st.slider(f"Select Nearest Distance for a Hospital (km)", float(1), float(5), 0.01)
]
Y["National School"] = [
    st.slider(f"Select Nearest Distance for a National School (km)", float(1), float(5), 0.01)
]
Y["Railway Station"] = [
    st.slider(f"Select Nearest Distance for a Railway Station (km)", float(1), float(5), 0.01)
]
Y["Bus Stop"] = [
    st.slider(f"Select Nearest Distance for a Bus Stop (km)", float(1), float(5), 0.01)
]
Y["Main Road"] = [
    st.slider(f"Select Nearest Distance for a Main Road (Meters)", float(1), float(2500), 0.01)
]

df = pd.DataFrame(Y)

if st.button("Predict"):
    if not selected_city:
        st.warning("Please Select the City")
    else:
        # Use the stored city value for prediction
        prediction = loaded_model.predict(df)
        rounded_prediction = (
            round(prediction[0] / 1000) * 1000
        )  # Round to the nearest 1000
        st.write(f"Ideal Value for your Land : {rounded_prediction}")
        st.write(
            f"Market value for your land : {rounded_prediction - rounded_prediction*0.5} to {rounded_prediction + rounded_prediction*0.5}"
        )

time_series_df = pd.read_csv("7_price_filtered.csv")

time_series_df['main_city'] = time_series_df['main_city'].str.lower()
time_series_df['posted_date'] = pd.to_datetime(time_series_df['posted_date'])

time_series_df['year'] = time_series_df['posted_date'].dt.year
time_series_df['quarter'] = time_series_df['posted_date'].dt.quarter

st.set_option("deprecation.showPyplotGlobalUse", False)

if st.button("See your land price in next 3 years"):
    st.subheader("Land Price Forecast for the Next 3 Years")

    train_data = time_series_df[time_series_df['posted_date'] < '2023-09-01']

    prediction_period = pd.date_range(start='2024-01-01', end='2027-03-31', freq='Q')
    prediction_data = pd.DataFrame({'posted_date': prediction_period, 'quarter': prediction_period.quarter})
    
    user_city = selected_city

    city_df = time_series_df[time_series_df['main_city'] == user_city]

    city_df.set_index('posted_date', inplace=True)
    quarterly_avg = city_df['price_land_pp'].resample('Q').mean()

    train_size = int(len(quarterly_avg) * 0.8)
    train, test = quarterly_avg[:train_size], quarterly_avg[train_size:]

    order = (1, 1, 1)
    seasonal_order = (1, 1, 1, 8)
    model = SARIMAX(train, order=order, seasonal_order=seasonal_order, enforce_stationarity=False)
    model_fit = model.fit()

    forecast_steps = len(test) + 12 
    forecast = model_fit.get_forecast(steps=forecast_steps)

    end_of_train_index = quarterly_avg.index[-1]

    forecast_index = pd.date_range(start='2024-01-01', periods=forecast_steps, freq='Q')

    smoothed_forecast = forecast.predicted_mean.rolling(window=5, min_periods=1).mean() 
    smoothed_forecast = smoothed_forecast.shift(-2)

    # Create a Plotly figure
    fig = px.line(x=forecast_index, y=smoothed_forecast, labels={'x': 'Time', 'y': 'Average Land Price'},
                title=f"Land Price Forecast for {user_city}")

    fig.update_layout(
        width=1400,
        height=700,
        title=dict(
            text=f"Land Price Forecast for {user_city}",
            x=0.5,  # Centered
            y=0.95,  # Adjust as needed
            xanchor='center',
            #yanchor='top',
            font=dict(size=23), automargin=True
        )
    )

    # Update line properties
    fig.update_traces(line=dict(color='red', dash='dash'))
    st.plotly_chart(fig, use_container_width=True)


