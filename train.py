"""
Indian House Price Prediction - Training Script
Trains Linear Regression, Random Forest, and Gradient Boosting models.
Saves the best model as model.pkl
"""

import pandas as pd
import numpy as np
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ─────────────────────────────────────────────────────────────────────────────
# 1. Load Data
# ─────────────────────────────────────────────────────────────────────────────
DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "indian_housing.csv")
df = pd.read_csv(DATA_PATH)
print(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")

# ─────────────────────────────────────────────────────────────────────────────
# 2. Basic Cleaning
# ─────────────────────────────────────────────────────────────────────────────
# Drop duplicates
df.drop_duplicates(inplace=True)

# Drop rows with missing Price or Area
df.dropna(subset=['Price', 'Area'], inplace=True)

# Remove extreme outliers (top/bottom 1%)
q_low  = df['Price'].quantile(0.01)
q_high = df['Price'].quantile(0.99)
df = df[(df['Price'] >= q_low) & (df['Price'] <= q_high)]

q_area_high = df['Area'].quantile(0.99)
df = df[df['Area'] <= q_area_high]

print(f"After cleaning: {df.shape[0]} rows")

# ─────────────────────────────────────────────────────────────────────────────
# 3. Feature Engineering
# ─────────────────────────────────────────────────────────────────────────────
df['price_per_sqft'] = df['Price'] / df['Area']

# Amenity score: sum of all binary amenity columns
amenity_cols = [
    'MaintenanceStaff', 'Gymnasium', 'SwimmingPool', 'LandscapedGardens',
    'JoggingTrack', 'RainWaterHarvesting', 'IndoorGames', 'ShoppingMall',
    'Intercom', 'SportsFacility', 'ATM', 'ClubHouse', 'School',
    '24X7Security', 'PowerBackup', 'CarParking', 'StaffQuarter', 'Cafeteria',
    'MultipurposeRoom', 'Hospital', 'WashingMachine', 'Gasconnection', 'AC',
    'Wifi', "Children'splayarea", 'LiftAvailable', 'BED', 'VaastuCompliant',
    'Microwave', 'GolfCourse', 'TV', 'DiningTable', 'Sofa', 'Wardrobe',
    'Refrigerator'
]
# Only use amenity cols that exist in the dataframe
amenity_cols = [c for c in amenity_cols if c in df.columns]
df['amenity_score'] = df[amenity_cols].sum(axis=1)

# Encode City
city_encoder = LabelEncoder()
df['City_encoded'] = city_encoder.fit_transform(df['City'])

# Encode Location (top 50 locations, rest as 'Other')
top_locations = df['Location'].value_counts().nlargest(50).index.tolist()
df['Location_clean'] = df['Location'].apply(lambda x: x if x in top_locations else 'Other')
loc_encoder = LabelEncoder()
df['Location_encoded'] = loc_encoder.fit_transform(df['Location_clean'])

# ─────────────────────────────────────────────────────────────────────────────
# 4. Feature Selection
# ─────────────────────────────────────────────────────────────────────────────
FEATURES = [
    'Area',
    'No. of Bedrooms',
    'Resale',
    'amenity_score',
    'City_encoded',
    'Location_encoded',
] + amenity_cols

TARGET = 'Price'

X = df[FEATURES]
y = np.log1p(df[TARGET])   # log-transform for better distribution

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")

# ─────────────────────────────────────────────────────────────────────────────
# 5. Train Models
# ─────────────────────────────────────────────────────────────────────────────
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest":     RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=200, learning_rate=0.1,
                                                    max_depth=5, random_state=42),
}

results = {}
trained_models = {}

print("\n" + "="*60)
print("Model Comparison")
print("="*60)

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Convert back from log scale
    y_test_orig = np.expm1(y_test)
    y_pred_orig = np.expm1(y_pred)

    mae  = mean_absolute_error(y_test_orig, y_pred_orig)
    rmse = np.sqrt(mean_squared_error(y_test_orig, y_pred_orig))
    r2   = r2_score(y_test_orig, y_pred_orig)

    results[name] = {"MAE": mae, "RMSE": rmse, "R2": r2}
    trained_models[name] = model

    print(f"\n{name}:")
    print(f"  MAE  : ₹{mae:,.0f}")
    print(f"  RMSE : ₹{rmse:,.0f}")
    print(f"  R²   : {r2:.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# 6. Select Best Model (highest R²)
# ─────────────────────────────────────────────────────────────────────────────
best_name = max(results, key=lambda k: results[k]["R2"])
best_model = trained_models[best_name]
print(f"\n✅ Best Model: {best_name} (R² = {results[best_name]['R2']:.4f})")

# ─────────────────────────────────────────────────────────────────────────────
# 7. Save Model + Encoders
# ─────────────────────────────────────────────────────────────────────────────
model_data = {
    "model":          best_model,
    "model_name":     best_name,
    "features":       FEATURES,
    "amenity_cols":   amenity_cols,
    "city_encoder":   city_encoder,
    "loc_encoder":    loc_encoder,
    "top_locations":  top_locations,
    "results":        results,
}

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")
with open(MODEL_PATH, "wb") as f:
    pickle.dump(model_data, f)

print(f"\n✅ Model saved to: {MODEL_PATH}")
print("Training complete!")
