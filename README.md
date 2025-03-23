# 🚗 Automobile Price Prediction & Visualization

This repository presents a complete pipeline for understanding, analyzing, and predicting automobile prices using machine learning. It includes comprehensive data analysis, interactive visual dashboards, model training, and a user-friendly **Streamlit** web app for price prediction.

---

## 📌 Project Motivation

This project was inspired by a business use-case: acquiring a new automobile rental service in the 1980s with a catalog of over 4,000 vehicles. The goal was to:

- Understand the vehicle catalog and categorize automobiles based on client types (e.g. family cars, sports cars).
- Analyze attributes like body type, fuel usage, power, and consumption to extract insights.
- Predict prices using machine learning to enable competitive re-pricing of vehicles.
- Help attract new clients by optimizing offerings based on market insights.

---

## 🗂️ Repository Structure

- **`auto_price.py`**  
  Main Streamlit application for car price prediction and data exploration.

- **`data_prep.ipynb`**  
  Notebook for data preprocessing, cleaning, and encoding.

- **`price_analysis_prediction.ipynb`**  
  Notebook for in-depth EDA, feature selection, model training, and evaluation.

- **`encoders.pkl`**  
  Pre-trained encoders for transforming categorical variables.

- **`model.pkl`**  
  Trained regression model used for predicting automobile prices.

- **`Auto_Dashboards.pdf`**  
  PDF report showing dashboard visualizations and data insights.

- **`data/`**  
  Contains raw dataset used in the analysis.

- **`pages/`**  
  Extra pages used in the Streamlit web app, including model explainability and statistical summaries.

---

## 🧠 What’s Being Done

### 🔍 Data Analysis
- Performed using Tableau (summarized in `Auto_Dashboards.pdf`).
- Explored relationships between car attributes like fuel type, power, consumption, and price.
- Identified surprising patterns such as inconsistent consumption vs. cylinder count.

### 🤖 Model Training
- Cleaned and encoded the dataset for ML modeling.
- Initial models: Linear Regression and Random Forest.
- Feature importance evaluated using SHAP values.
- Dimensionality reduction applied to eliminate misleading encoded variables.
- Final model trained with selected impactful features.

### 🌐 Streamlit Web App
- Predicts car prices based on user inputs.
- Features three tabs:
  - **Predictor** – Enter car features and get price predictions.
  - **Stats** – Summary stats and visualizations.
  - **Explainability** – SHAP-based model explainability.

---

## 🚀 How to Run the Streamlit App

To launch the price prediction web application, make sure that you have the environment setup completed and all the dependecies' requirements are met. Finally, run the following command in the terminal:

    streamlit run auto_price.py

This will start the app on http://localhost:8501.


## 📝 Notes

Make sure encoders.pkl and model.pkl are in the root directory — they are required for the app to function.

Extra insights are available via the sidebar pages of the app.

The SHAP explainability plots help users understand the influence of each feature.

The app lets users interactively enter vehicle details and returns an estimated price based on trained ML models.
