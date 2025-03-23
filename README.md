# ML-viz 🚗📊

This repository provides a full pipeline for automobile price prediction using machine learning, along with interactive visualizations and dashboards. It includes data preparation, exploratory data analysis (EDA), model training, and a user-friendly **Streamlit** web app to predict car prices based on input features.

---

## 📁 Repository Contents

- `auto_price.py`  
  → Main **Streamlit app** for predicting automobile prices.

- `data_prep.ipynb`  
  → Jupyter Notebook for data cleaning, preprocessing, and feature engineering.

- `price_analysis_prediction.ipynb`  
  → Notebook for visual analysis and training machine learning models.

- `encoders.pkl`  
  → Pretrained encoders used for transforming categorical features.

- `model.pkl`  
  → Trained ML model used for price prediction.

- `Auto_Dashboards.pdf`  
  → PDF dashboards summarizing key insights from the dataset.

- `data/`  
  → Contains raw dataset(s) used for training and testing.

- `pages/`  
  → Additional pages/components for the Streamlit app.

---

## 🚀 Streamlit App

To launch the price prediction web application, make sure that you have the environment setup completed and all the dependecies' requirements are met. Finally, run the following command in the terminal:

    * streamlit run auto_price.py


## 📝 Notes
Make sure encoders.pkl and model.pkl are in the root directory — they are required for the app to function.

You can explore additional data or insights in the provided notebooks.

Extra Streamlit pages (if any) are located in the pages/ directory and will appear in the sidebar of the app.

The app lets users interactively enter vehicle details and returns an estimated price based on trained ML models.
