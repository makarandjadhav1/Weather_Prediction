import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib
import os
import plotly.express as px
import uuid

# Set page config
st.set_page_config(
    page_title="Weather Prediction App",
    page_icon="🌤️",
    layout="wide"
)

# Title and description
st.title("🌤️ Weather Prediction App")
st.write("Upload historical weather data (CSV) to train a model and predict tomorrow's temperature. Ensure the last column is the target (temperature) and other columns are features.")

# Initialize session state
if 'trained_model' not in st.session_state:
    st.session_state.trained_model = None
    st.session_state.scaler = None
    st.session_state.label_encoders = None
    st.session_state.feature_columns = None
    st.session_state.last_data = None
    st.session_state.session_id = str(uuid.uuid4())

# Function to reset session state
def reset_session():
    st.session_state.trained_model = None
    st.session_state.scaler = None
    st.session_state.label_encoders = None
    st.session_state.feature_columns = None
    st.session_state.last_data = None
    st.session_state.session_id = str(uuid.uuid4())

# File uploader section
st.header("Step 1: Train Your Model")
uploaded_file = st.file_uploader("Upload historical weather data CSV", type="csv", key=st.session_state.session_id)

if uploaded_file is not None:
    try:
        # Read the CSV file
        df = pd.read_csv(uploaded_file)
        
        # Handle missing values
        df = df.fillna(df.mean(numeric_only=True))
        df = df.fillna(df.mode().iloc[0])
        
        # Display the first few rows
        st.subheader("Preview of Your Data")
        st.dataframe(df.head())
        
        # Validate data
        if df.shape[1] < 2:
            st.error("CSV must have at least one feature and one target column.")
            st.stop()
        if len(df) < 10:
            st.error("Dataset is too small. Please provide at least 10 rows.")
            st.stop()
        if not np.issubdtype(df.iloc[:, -1].dtype, np.number):
            st.error("Target column (last column) must be numerical (e.g., temperature).")
            st.stop()
        
        # Feature engineering
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            df['day_of_year'] = df['date'].dt.dayofyear
            df['month'] = df['date'].dt.month
        
        # Prepare features and target
        X = df.drop(columns=[df.columns[-1]] + (['date'] if 'date' in df.columns else []))
        y = df.iloc[:, -1]
        
        # Store feature columns
        st.session_state.feature_columns = X.columns.tolist()
        
        # Handle categorical variables
        categorical_columns = X.select_dtypes(include=['object']).columns
        label_encoders = {}
        for column in categorical_columns:
            label_encoders[column] = LabelEncoder()
            X[column] = label_encoders[column].fit_transform(X[column])
        st.session_state.label_encoders = label_encoders
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale the features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        st.session_state.scaler = scaler
        
        # Train the model
        model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
        model.fit(X_train_scaled, y_train)
        st.session_state.trained_model = model
        
        # Store the last row for prediction
        st.session_state.last_data = df.iloc[-1].to_dict()
        
        # Save model (optional)
        joblib.dump(model, 'weather_model.pkl')
        joblib.dump(scaler, 'scaler.pkl')
        joblib.dump(label_encoders, 'label_encoders.pkl')
        
        # Model performance
        train_score = model.score(X_train_scaled, y_train)
        test_score = model.score(X_test_scaled, y_test)
        y_pred = model.predict(X_test_scaled)
        rmse = mean_squared_error(y_test, y_pred, squared=False)
        mae = mean_absolute_error(y_test, y_pred)
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
        
        st.subheader("Model Training Results")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Training Score", f"{train_score:.2%}")
        with col2:
            st.metric("Testing Score", f"{test_score:.2%}")
        with col3:
            st.metric("RMSE", f"{rmse:.2f}")
        with col4:
            st.metric("MAE", f"{mae:.2f}")
        st.write(f"Cross-Validation Score: {cv_scores.mean():.2%} ± {cv_scores.std():.2%}")
        
        # Feature importance
        st.subheader("Feature Importance")
        feature_importance = pd.DataFrame({
            'Feature': X.columns,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        fig = px.bar(feature_importance,
                     x='Feature',
                     y='Importance',
                     title='Feature Importance in Weather Prediction',
                     color='Importance',
                     color_continuous_scale='Viridis')
        fig.update_layout(xaxis_title="Features", yaxis_title="Importance", xaxis_tickangle=45)
        st.plotly_chart(fig, use_container_width=True)
        
        st.success("Model trained successfully! You can now make predictions.")
        
        # Reset button
        if st.button("Reset and Train New Model"):
            reset_session()
            st.experimental_rerun()
        
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.write("Ensure your CSV has the correct format: features + numerical target column (temperature).")

# Prediction section
st.header("Step 2: Predict Tomorrow's Weather")

if st.session_state.trained_model is not None:
    # Display last data
    if st.session_state.last_data is not None:
        st.subheader("Last Data Used for Prediction")
        last_data_df = pd.DataFrame([st.session_state.last_data])
        st.dataframe(last_data_df)
    
    # Create input form
    st.subheader("Enter Tomorrow's Weather Features")
    input_data = {}
    cols = st.columns(3)
    
    for i, column in enumerate(st.session_state.feature_columns):
        col = cols[i % 3]
        if column in st.session_state.label_encoders:
            categories = st.session_state.label_encoders[column].classes_
            default_value = st.session_state.last_data.get(column, categories[0])
            try:
                default_index = list(categories).index(default_value)
            except ValueError:
                default_index = 0
            selected = col.selectbox(f"{column}", options=categories, index=default_index)
            input_data[column] = selected
        else:
            min_val = float(X[column].min()) if 'X' in locals() else -1000.0
            max_val = float(X[column].max()) if 'X' in locals() else 1000.0
            default_val = float(st.session_state.last_data.get(column, X[column].median() if 'X' in locals() else 0.0))
            value = col.number_input(f"{column}", min_value=min_val, max_value=max_val, value=default_val)
            input_data[column] = value
    
    if st.button("Predict Tomorrow's Weather"):
        try:
            # Prepare input data
            input_df = pd.DataFrame([input_data])
            
            # Encode categorical variables
            for column in st.session_state.label_encoders:
                input_df[column] = st.session_state.label_encoders[column].transform(input_df[column])
            
            # Scale features
            input_scaled = st.session_state.scaler.transform(input_df)
            
            # Make prediction
            prediction = st.session_state.trained_model.predict(input_scaled)[0]
            
            # Display prediction
            st.subheader("Prediction Result")
            st.success(f"Predicted Temperature for Tomorrow: {prediction:.2f}°C")
            
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")
            st.write("Ensure input values match the training data format.")
else:
    st.warning("Please train a model first by uploading historical data above.")