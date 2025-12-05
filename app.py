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

# Cashing data
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

# st.sidebar.header("Navigation")
# page = st.sidebar.selectbox("Go to", ["Data Exploration", "Model Training", "Model Evaluation"])

# Nav pannel
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Data Exploration",
    "⚙️ Model Training",
    "📈 Model Evaluation",
    "🧮 Live Prediction"
])

# Data Exploration
with tab1:
    st.header("Dataset Overview")
    
    st.write("Shape:", df.shape)
    st.write(df.head())
    
    st.subheader("Summary Statistics")
    st.write(df.describe())
    
    st.subheader("Histogram")
    numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns
    col = st.selectbox("Select feature for histogram:", numeric_cols)
    
    st.bar_chart(df[col].dropna())

    # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

    st.subheader("Correlation Heatmap")

    import matplotlib.pyplot as plt
    import seaborn as sns
    
    df_numeric = df.select_dtypes(include=['float64', 'int64'])

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(df_numeric.corr(), cmap="coolwarm", annot=False)
    st.pyplot(fig)

    # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

    st.subheader("Scatter Plot")

    x_feature = st.selectbox("Select X-axis:", df_numeric.columns)
    y_feature = st.selectbox("Select Y-axis:", df_numeric.columns)

    fig, ax = plt.subplots()
    ax.scatter(df_numeric[x_feature], df[y_feature], alpha=0.5)
    ax.set_xlabel(x_feature)
    ax.set_ylabel(y_feature)
    st.pyplot(fig)

    # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

    st.subheader("Boxplot")

    col = st.selectbox("Select feature:", df_numeric.columns)
    fig, ax = plt.subplots()
    sns.boxplot(df_numeric[col], ax=ax)
    st.pyplot(fig)

# Model Training
with tab2:
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
            filename = "models/lr_model.pkl"

        elif model_choice == "Random Forest":
            model = RandomForestRegressor(n_estimators=200, random_state=42)
            filename = "models/rf_model.pkl"

        else:
            model = SVR(kernel="rbf", C=10, epsilon=0.1)
            filename = "models/svr_model.pkl"

        model.fit(X_train, y_train)

        with open(filename, "wb") as f:
            pickle.dump(model, f)

        st.success(f"{model_choice} saved as {filename}")

# Model Evaluation
with tab3:
    st.subheader("Choose Model to Evaluate")

    model_file = st.selectbox(
        "Select model file:",
        ["models/lr_model.pkl", "models/rf_model.pkl", "models/svr_model.pkl"]
    )

    try:
        with open(model_file, "rb") as f:
            model = pickle.load(f) 
    except:
        st.error("No trained model found. Please train a model first.")
        st.stop()

    # Re-create preprocessing pipeline (must match training)
    df_eval = pd.read_csv("dataset/AirQualityUCI.csv", sep=';',  decimal=',', na_values=-200)
    df_eval = df_eval.dropna(axis=1, how='all')
    df_eval = df_eval.dropna(how="all")

    df_eval = df_eval.rename(columns={
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

    df_eval['Time'] = df_eval['Time'].str.replace('.', ':', regex=False)
    df_eval['Time'] = df_eval['Time'].apply(lambda x: x + ':00' if len(x) == 5 else x)

    df_eval['Datetime'] = pd.to_datetime(
        df_eval['Date'] + ' ' + df_eval['Time'],
        format='%d/%m/%Y %H:%M:%S',
        errors='coerce'
    )

    df_eval['Hour'] = df_eval['Datetime'].dt.hour
    df_eval['Month'] = df_eval['Datetime'].dt.month
    df_eval['DayOfWeek'] = df_eval['Datetime'].dt.dayofweek

    df_eval = df_eval.drop(columns=['Date', 'Time', 'Datetime'])

    num_cols = df_eval.select_dtypes(include=['float64', 'int64']).columns
    df_eval[num_cols] = df_eval[num_cols].fillna(df_eval[num_cols].median())

    target = "Benzene"
    X_eval = df_eval.drop(columns=[target])
    y_eval = df_eval[target]

    # Make predictions
    y_pred = model.predict(X_eval)

    st.subheader("Regression Performance Metrics")
    st.write("**MAE:**", mean_absolute_error(y_eval, y_pred))
    st.write("**MSE:**", mean_squared_error(y_eval, y_pred))
    st.write("**RMSE:**", np.sqrt(mean_squared_error(y_eval, y_pred)))
    st.write("**R² Score:**", r2_score(y_eval, y_pred))

# Live Prediction
with tab4:
    st.header("Live Prediction")

    try:
        with open("models/rf_model.pkl", "rb") as f:
            model = pickle.load(f)
    except:
        st.error("Train and save a model first!")
        st.stop()

    # Dynamic numeric inputs
    st.subheader("Enter sensor measurements:")

    inputs = {}
    for col in X.columns:  # X is your final feature set
        inputs[col] = st.number_input(
            col,
            min_value=float(df[col].min()),
            max_value=float(df[col].max()),
            value=float(df[col].median())
        )

    input_df = pd.DataFrame([inputs])

    if st.button("Predict"):
        pred = model.predict(input_df)[0]
        st.success(f"Predicted Benzene Concentration (C6H6): **{pred:.2f} µg/m³**")
