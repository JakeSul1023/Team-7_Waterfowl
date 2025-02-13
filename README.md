🦆 Team-7 Waterfowl Migration Prediction System
Overview

This repository is the core backend and machine learning system for the Waterfowl Migration Prediction System. It processes historical migration data from the TTU Wildlife Department, integrates NOAA weather data, and generates predicted migration patterns for waterfowl across the Mississippi Flyway.

The system will use machine learning models to forecast duck movements up to 10 days in advance, leveraging real-time weather data and historical trends. The predicted migration data is stored in a static CSV file, which can be used for visualization in the frontend repository (Duck Data repo).

How This Repo Fits into the Project

    🖥 Handles all machine learning and data processing
    📊 Generates CSV files with predicted waterfowl migration paths
    🌎 Integrates with NOAA weather data for dynamic forecasting
    🔗 Outputs data for frontend visualization in the Duck Data repository

Current Features

    ✅ Data Preprocessing Pipeline – Cleans and organizes historical migration data
    ✅ CSV Data Export – Stores structured migration data for visualization
    ✅ NOAA Weather Data Integration – Prepares real-time weather conditions for ML models

Upcoming Features

    🚀 Machine Learning Model – Develop a predictive model for forecasting migration trends up to 10 days ahead
    🌦 Dynamic Weather-Based Predictions – Adjust forecasts based on real-time NOAA weather data
    🖥 HPC Optimization – Deploy the model on TTU’s High-Performance Computing (HPC) system for large-scale predictions
    🔗 Frontend Integration – Connect with the Duck Data repo for heatmap visualization

How It Will Work

    1️⃣ Historical Data Processing – Prepares and cleans migration data from TTU Wildlife Department
    2️⃣ Weather-Based Predictions – Runs ML models incorporating current and forecasted weather conditions
    3️⃣ CSV Output Generation – Exports predicted migration paths as a structured dataset
    4️⃣ Frontend Integration – The processed data is sent to the Duck Data repository for visualization

Project Details

    Client: Dr. Cohen, TTU Wildlife Department
    Data Sources: TTU Wildlife Department (Migration Data), NOAA (Weather Data)
    Tech Stack: Python (ML), R (Data Processing), HPC (Model Training), CSV Export

Future Considerations

    🔹 Improve ML model accuracy based on species-specific migration behaviors
    🔹 Implement real-time data updates for adaptive forecasting
    🔹 Explore long-term forecasting beyond the 10-day window
