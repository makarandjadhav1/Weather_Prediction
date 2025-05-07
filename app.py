import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import os
import plotly.express as px

# Set page config
st.set_page_config(
    page_title="Weather Prediction App",
    page_icon="🌤️",
    layout="wide"
)

# Title and description
st.title("🌤️ Weather Prediction App")
st.write("Upload historical weather data to train a model, then predict tomorrow's weather!")

# Initialize session state for model persistence
if 'trained_model' not in st.session_state:
    st.session_state.trained_model = None
    st.session_state.scaler = None
    st.session_state.label_encoders = None
    st.session_state.feature_columns = None

# File uploader section
st.header("Step 1: Train Your Model")
uploaded_file = st.file_uploader("Upload historical weather data CSV", type="csv")

if uploaded_file is not None:
    try:
        # Read the CSV file
        df = pd.read_csv(uploaded_file)
        
        # Display the first few rows of the data
        st.subheader("Preview of your data:")
        st.dataframe(df.head())
        
        # Check if we have enough data
        if len(df) < 2:
            st.error("The dataset is too small. Please provide more data.")
        else:
            # Prepare features and target
            # Assuming the last column is the target (temperature)
            X = df.iloc[:, :-1]
            y = df.iloc[:, -1]
            
            # Store feature columns for later use
            st.session_state.feature_columns = X.columns.tolist()
            
            # Handle categorical variables
            categorical_columns = X.select_dtypes(include=['object']).columns
            label_encoders = {}
            
            for column in categorical_columns:
                label_encoders[column] = LabelEncoder()
                X[column] = label_encoders[column].fit_transform(X[column])
            
            # Store label encoders in session state
            st.session_state.label_encoders = label_encoders
            
            # Split the data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Scale the features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Store scaler in session state
            st.session_state.scaler = scaler
            
            # Train the model
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train_scaled, y_train)
            
            # Store model in session state
            st.session_state.trained_model = model
            
            # Display model performance
            train_score = model.score(X_train_scaled, y_train)
            test_score = model.score(X_test_scaled, y_test)
            
            st.subheader("Model Training Results")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Training Score", f"{train_score:.2%}")
            with col2:
                st.metric("Testing Score", f"{test_score:.2%}")
            
            # Feature importance
            st.subheader("Feature Importance")
            feature_importance = pd.DataFrame({
                'Feature': X.columns,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            # Create a bar chart using plotly
            fig = px.bar(feature_importance, 
                         x='Feature', 
                         y='Importance',
                         title='Feature Importance in Weather Prediction')
            st.plotly_chart(fig)
            
            st.success("Model trained successfully! You can now make predictions.")
            
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.write("Please make sure your CSV file has the correct format.")

# Prediction section
st.header("Step 2: Predict Tomorrow's Weather")

if st.session_state.trained_model is not None:
    # Create input form for tomorrow's weather features
    st.subheader("Enter Tomorrow's Weather Features")
    
    input_data = {}
    cols = st.columns(3)  # Create 3 columns for better layout
    
    for i, column in enumerate(st.session_state.feature_columns):
        # Determine which column to place the input in (spread across 3 columns)
        col = cols[i % 3]
        
        # Check if the feature is categorical
        if column in st.session_state.label_encoders:
            # Get the original categories before encoding
            categories = st.session_state.label_encoders[column].classes_
            selected = col.selectbox(f"{column}", options=categories)
            input_data[column] = selected
        else:
            # For numerical features, get min/max from training data to set reasonable bounds
            min_val = float(X[column].min())
            max_val = float(X[column].max())
            default_val = float(X[column].median())
            value = col.number_input(f"{column}", min_value=min_val, max_value=max_val, value=default_val)
            input_data[column] = value
    
    if st.button("Predict Tomorrow's Weather"):
        try:
            # Prepare the input data for prediction
            input_df = pd.DataFrame([input_data])
            
            # Encode categorical variables
            for column in st.session_state.label_encoders:
                input_df[column] = st.session_state.label_encoders[column].transform(input_df[column])
            
            # Scale the features
            input_scaled = st.session_state.scaler.transform(input_df)
            
            # Make prediction
            prediction = st.session_state.trained_model.predict(input_scaled)[0]
            
            # Display prediction
            st.subheader("Prediction Result")
            st.success(f"Predicted Temperature for Tomorrow: {prediction:.2f}°C")
            
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")
else:
    st.warning("Please train a model first by uploading historical data above.")