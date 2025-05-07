import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os
import plotly.express as px

# Set page config - must be the first Streamlit command
st.set_page_config(
    page_title="Weather Prediction App",
    page_icon="🌤️",
    layout="wide"
)

# Title and description
st.title("🌤️ Weather Prediction App")
st.write("Upload your weather data CSV file to predict tomorrow's weather!")

# File uploader
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

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
            
            # Split the data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Scale the features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train the model
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train_scaled, y_train)
            
            # Make prediction for tomorrow
            # Use the last row of data as input for tomorrow's prediction
            tomorrow_data = scaler.transform(X.iloc[[-1]])
            prediction = model.predict(tomorrow_data)[0]
            
            # Display prediction
            st.subheader("Tomorrow's Weather Prediction")
            st.write(f"Predicted Temperature: {prediction:.2f}°C")
            
            # Display model performance
            train_score = model.score(X_train_scaled, y_train)
            test_score = model.score(X_test_scaled, y_test)
            
            st.subheader("Model Performance")
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
            
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.write("Please make sure your CSV file has the correct format with numerical data.")
else:
    st.info("Please upload a CSV file to get started.")