# 📞 Customer Churn Risk Predictor

This is an interactive web app built with Streamlit and Scikit-learn to predict customer churn risk based on their account and service details.



## 🎯 Project Overview

This project demonstrates a full end-to-end data science workflow:
1.  **Data Analysis & Modeling:** Trained an XGBoost classifier on the Telco Churn dataset (see `Model_Training.ipynb`).
2.  **Preprocessing:** Built a `ColumnTransformer` pipeline to handle mixed numeric and categorical data.
3.  **App Development:** Created an interactive front-end using Streamlit.
4.  **Deployment:** Deployed the app via Streamlit Community Cloud.

## 🚀 How to Run Locally

1. Clone the repository:
   ```bash
   git clone https://github.com/adhiraj0905/customer-churn.git
   cd churn-predictor