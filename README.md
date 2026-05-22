# Traffic Flow Analysis and Prediction

This project analyzes traffic flow patterns using vehicle count data and builds a machine learning model to predict traffic conditions. The project includes data exploration, model training, and an interactive Streamlit web application for displaying prediction results.

## Project Overview

Traffic congestion is an important issue in urban areas. This project uses a traffic dataset containing vehicle counts for different vehicle types, including cars, bikes, buses, and trucks. The dataset also includes time, date, day of the week, total vehicle count, and traffic situation labels.

The goal of this project is to understand traffic patterns and predict traffic situations based on vehicle count and time-related features.

## Features

- Data cleaning and preprocessing
- Exploratory data analysis using Python
- Vehicle count analysis by time and day
- Machine learning model for traffic situation prediction
- Label encoding for categorical variables
- Streamlit web app for interactive prediction
- Saved model files for reuse

## Dataset

The dataset contains traffic information collected at regular time intervals. Main columns include:

- Time
- Date
- Day of the week
- Car count
- Bike count
- Bus count
- Truck count
- Total vehicle count
- Traffic situation

The traffic situation is categorized into different levels such as heavy, high, normal, and low traffic.

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Streamlit
- Jupyter Notebook

## Project Files

```text
Traffic Prediction Dataset.ipynb   # Data analysis and exploration notebook
Traffic.csv                        # Traffic dataset
train_model.py                     # Model training script
app.py                             # Streamlit web application
traffic_model.pkl                  # Saved machine learning model
label_encoder_day.pkl              # Saved encoder for day feature
label_encoder_target.pkl           # Saved encoder for target labels