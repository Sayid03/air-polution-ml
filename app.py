import streamlit as st
import pandas as pd
import numpy as np
import pickle

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.title("Air Quality ML Dashboard")

@st.cache_data
def load_data():
    df = pd.read_csv("dataset/AirQualityUCI.csv", sep=';',  decimal=',', na_values=-200)
    
    df = df.rename(columns={
    "CO(GT)": "CO",
    "C6H6(GT)": "Benzene",
    "NO2(GT)": "NO2",
    "NMHC(GT)": "NMHC",
    "PT08.S1(CO)": "Sensor_CO",
    "PT08.S2(NMHC)": "Sensor_NMHC",
    "PT08.S4(NO2)": "Sensor_NO2",
    "T": "Temperature",
    "RH": "Humidity",
    "AH": "AbsHumidity"
    })
    
    df = df.dropna(axis=1, how='all')
    df = df.dropna(how='all')

    return df

df = load_data()

st.sidebar.header("Navigation")
page = st.sidebar.selectbox("Go to", ["Data Exploration", "Model Training", "Model Evaluation"])

if page == "Data Exploration":
    st.header("Dataset Overview")
    
    st.write("Shape:", df.shape)
    st.write(df.head())
    
    st.subheader("Summary Statistics")
    st.write(df.describe())
    
    st.subheader("Histogram Example")
    numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns
    col = st.selectbox("Select feature for histogram:", numeric_cols)
    
    st.bar_chart(df[col].dropna())

elif page == "Model Training":
    st.header("Train Models")

    st.write("Preparing the data...")

    df['Time'] = df['Time'].str.replace('.', ':', regex=False)
    df['Time'] = df['Time'].apply(lambda x: x + ':00' if len(x) == 5 else x)
    
    df['Datetime'] = pd.to_datetime(
        df['Date'] + ' ' + df['Time'],
        format='%d/%m/%Y %H:%M:%S',
        errors='coerce'
    )
    
    df['Hour'] = df['Datetime'].dt.hour
    df['Month'] = df['Datetime'].dt.month
    df['DayOfWeek'] = df['Datetime'].dt.dayofweek
    
    df = df.drop(columns=['Date', 'Time', 'Datetime'])
    
    num_cols = df.select_dtypes(include=['float64', 'int64']).columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())

    target = 'Benzene'
    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    model_choice = st.selectbox("Choose model", ["Linear Regression", "Random Forest", "SVR"])

    if st.button("Train Model"):
        if model_choice == "Linear Regression":
            model = LinearRegression()
        elif model_choice == "Random Forest":
            model = RandomForestRegressor(n_estimators=200, random_state=42)
        else:
            model = SVR(kernel='rbf', C=10, epsilon=0.1)

        model.fit(X_train, y_train)

        # Save model
        with open("trained_model.pkl", "wb") as f:
            pickle.dump(model, f)

        st.success("Model trained and saved!")

elif page == "Model Evaluation":
    st.header("Evaluate Trained Model")

    uploaded_model = None

    try:
        with open("trained_model.pkl", "rb") as f:
            uploaded_model = pickle.load(f)
    except:
        st.error("No model found. Train a model first.")
    
    if uploaded_model:
        y_pred = uploaded_model.predict(X_test)

        st.subheader("Performance Metrics")
        st.write("MAE:", mean_absolute_error(y_test, y_pred))
        st.write("MSE:", mean_squared_error(y_test, y_pred))
        st.write("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
        st.write("R²:", r2_score(y_test, y_pred))
