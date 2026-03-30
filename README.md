# 🍽️ Food Bill Prediction & Data Modeling App

## 🚀 Live Application

🔗 Streamlit Deployment:  
    https://food-data-modeling-wn8unov3ydga96fwanz3fe.streamlit.app

---

## 📌 Project Overview

This project is a Machine Learning-based Food Bill Prediction system built using a messy real-world styled dataset.

The project demonstrates:

- Data cleaning and preprocessing
- Feature engineering using One-Hot Encoding
- Linear Regression model building
- Model serialization using Pickle
- Deployment using Streamlit Cloud
- Basic chatbot-style UI integration

The application predicts the estimated food bill amount based on user inputs such as restaurant, cuisine type, meal time, rating, and delivery details.

---

## 📊 Dataset Information

Dataset File:
food_messy_dataset1.csv

Features include:
- Restaurant Name
- Cuisine Type
- Meal Time
- Rating
- Preparation Time
- Delivery Time
- Promo Code
- Order Value (Target Variable)

The dataset was initially messy and required preprocessing before modeling.

---

## 🧠 Machine Learning Model

Model Used:
- Linear Regression

Notebook:
liner_regression_model (2).ipynb

Steps Performed:
1. Data Cleaning
2. Handling Missing Values
3. One-Hot Encoding for categorical variables
4. Feature alignment to avoid mismatch errors
5. Model training
6. Model evaluation using R², MAE, MSE
7. Model saving using Pickle (model.pkl)

---

## 💻 Streamlit Application

Main File:
app.py

The Streamlit app allows users to:
- Select restaurant
- Choose cuisine type
- Select meal time
- Enter rating and delivery details
- Get predicted food bill instantly

Run locally using:

streamlit run app.py

---

## 🤖 Chatbot Interface

File:
food_bill_chatbot.html

Provides a simple chatbot-style interface for interacting with the prediction system.

---

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Linear Regression
- Pickle
- Streamlit
- HTML
- Git & GitHub

---

## 📈 Model Evaluation Metrics

- R² Score
- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
