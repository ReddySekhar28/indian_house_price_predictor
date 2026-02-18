# Indian Smart House Price Prediction System

## 🏠 Project Overview
This project is a machine learning web application that predicts house prices in Indian cities. It uses a location-aware approach by integrating geocoding APIs to convert user-provided locations (City, Area) into geographic coordinates (Latitude, Longitude), which are then used as features in the prediction model.

## 🎯 Objectives
- Build a user-friendly interface for house price prediction.
- Implement location intelligence using Geocoding APIs.
- Compare multiple ML models (Linear Regression, Random Forest, Gradient Boosting) to find the best performer.
- Deploy the solution using Streamlit.

## 🛠 Tech Stack
- **Python**: Core programming language.
- **Pandas & NumPy**: Data manipulation and analysis.
- **Scikit-learn**: Machine learning model training and evaluation.
- **Geopy**: Geocoding and location intelligence.
- **Streamlit**: Web application framework.
- **Matplotlib & Seaborn**: Data visualization.

## 📂 Folder Structure
```
indian_house_price_project/
├── data/               # Dataset files
├── notebooks/          # Jupyter notebooks for EDA and training
├── train.py            # Training script
├── app.py              # Streamlit application
├── model.pkl           # Trained model (generated)
├── requirements.txt    # Project dependencies
└── README.md           # Project documentation
```

## 🚀 How to Run
1.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
2.  Run the Streamlit app:
    ```bash
    streamlit run app.py
    ```
