"""
Indian Smart House Price Prediction System
Streamlit Web Application with Location Intelligence
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import time

# ─────────────────────────────────────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Indian House Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 2.5rem 2rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    }

    .main-header h1 {
        color: #ffffff;
        font-size: 2.4rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.5px;
    }

    .main-header p {
        color: #a8b2d8;
        font-size: 1.05rem;
        margin-top: 0.5rem;
    }

    .prediction-card {
        background: linear-gradient(135deg, #0f3460 0%, #533483 100%);
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        box-shadow: 0 8px 32px rgba(83, 52, 131, 0.4);
        margin: 1.5rem 0;
        animation: fadeIn 0.5s ease-in;
    }

    .prediction-card h2 {
        color: #ffffff;
        font-size: 1.2rem;
        font-weight: 500;
        margin: 0 0 0.5rem 0;
        opacity: 0.85;
    }

    .prediction-card .price {
        color: #e2b96f;
        font-size: 2.8rem;
        font-weight: 700;
        margin: 0.5rem 0;
    }

    .prediction-card .sub {
        color: #a8b2d8;
        font-size: 0.95rem;
    }

    .metric-card {
        background: #1a1a2e;
        border: 1px solid #2a2a4e;
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
    }

    .metric-card .label {
        color: #a8b2d8;
        font-size: 0.85rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    .metric-card .value {
        color: #e2b96f;
        font-size: 1.4rem;
        font-weight: 700;
        margin-top: 0.3rem;
    }

    .section-header {
        color: #e2b96f;
        font-size: 1rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 1rem;
        border-bottom: 1px solid #2a2a4e;
        padding-bottom: 0.5rem;
    }

    .stButton > button {
        background: linear-gradient(135deg, #e2b96f, #c9973a) !important;
        color: #1a1a2e !important;
        font-weight: 700 !important;
        font-size: 1.1rem !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 0.8rem 2rem !important;
        width: 100% !important;
        transition: all 0.2s ease !important;
        box-shadow: 0 4px 15px rgba(226, 185, 111, 0.3) !important;
    }

    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(226, 185, 111, 0.5) !important;
    }

    .info-box {
        background: rgba(226, 185, 111, 0.1);
        border-left: 3px solid #e2b96f;
        border-radius: 0 8px 8px 0;
        padding: 0.8rem 1rem;
        margin: 0.5rem 0;
        font-size: 0.9rem;
        color: #c8d0e8;
    }

    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to   { opacity: 1; transform: translateY(0); }
    }

    .stSelectbox label, .stSlider label, .stNumberInput label, .stMultiSelect label {
        color: #a8b2d8 !important;
        font-weight: 500 !important;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Load Model
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), "model.pkl")
    with open(model_path, "rb") as f:
        return pickle.load(f)

model_data = load_model()
model        = model_data["model"]
features     = model_data["features"]
amenity_cols = model_data["amenity_cols"]
city_encoder = model_data["city_encoder"]
loc_encoder  = model_data["loc_encoder"]
top_locations = model_data["top_locations"]
results      = model_data["results"]
model_name   = model_data["model_name"]

# ─────────────────────────────────────────────────────────────────────────────
# Geocoding (Location Intelligence)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600)
def geocode_location(city: str, area: str = "") -> dict:
    """Convert city/area to lat/lon using Nominatim."""
    try:
        from geopy.geocoders import Nominatim
        from geopy.exc import GeocoderTimedOut
        geolocator = Nominatim(user_agent="indian_house_price_predictor_v1")
        query = f"{area}, {city}, India" if area else f"{city}, India"
        location = geolocator.geocode(query, timeout=5)
        if location:
            return {"lat": location.latitude, "lon": location.longitude,
                    "address": location.address, "success": True}
    except Exception:
        pass
    return {"lat": None, "lon": None, "address": None, "success": False}

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏠 About This App")
    st.markdown("""
    **Indian Smart House Price Predictor** uses machine learning to estimate
    property prices across 6 major Indian cities.

    **Cities Supported:**
    - 🏙️ Hyderabad
    - 🌆 Mumbai
    - 🌃 Bangalore
    - 🏛️ Delhi
    - 🌉 Chennai
    - 🌁 Kolkata
    """)

    st.markdown("---")
    st.markdown("## 🤖 Model Info")
    st.markdown(f"**Active Model:** `{model_name}`")

    st.markdown("### 📊 Model Comparison")
    for name, r in results.items():
        icon = "✅" if name == model_name else "  "
        st.markdown(f"{icon} **{name}**")
        st.caption(f"R² = {r['R2']:.3f} | MAE = ₹{r['MAE']/1e5:.1f}L")

    st.markdown("---")
    st.markdown("### 📁 Dataset Info")
    st.markdown("- **Source:** 6 Indian Cities\n- **Records:** ~33,000\n- **Features:** 40+")

# ─────────────────────────────────────────────────────────────────────────────
# Main Header
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🏠 Indian Smart House Price Predictor</h1>
    <p>Location-Aware ML System for Real Estate Valuation</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Input Form
# ─────────────────────────────────────────────────────────────────────────────
col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.markdown('<div class="section-header">📍 Location Details</div>', unsafe_allow_html=True)

    city = st.selectbox(
        "City",
        options=sorted(city_encoder.classes_.tolist()),
        index=0,
        help="Select the city where the property is located"
    )

    # Location dropdown based on city
    location_input = st.selectbox(
        "Area / Locality",
        options=["Other"] + sorted(top_locations),
        help="Select the area/locality. Choose 'Other' if not listed."
    )

    show_map = st.checkbox("📍 Show Location on Map", value=True)

    st.markdown('<div class="section-header">🏗️ Property Details</div>', unsafe_allow_html=True)

    sqft = st.number_input(
        "Property Size (sq ft)",
        min_value=300, max_value=10000,
        value=1200, step=50,
        help="Total built-up area in square feet"
    )

    bedrooms = st.selectbox(
        "Number of Bedrooms",
        options=[1, 2, 3, 4, 5, 6],
        index=1,
        help="Total number of bedrooms"
    )

    resale = st.radio(
        "Property Type",
        options=["New Property", "Resale Property"],
        horizontal=True
    )
    resale_val = 1 if resale == "Resale Property" else 0

with col_right:
    st.markdown('<div class="section-header">🏊 Amenities</div>', unsafe_allow_html=True)

    # Group amenities for better UX
    amenity_groups = {
        "🏋️ Fitness & Recreation": ["Gymnasium", "SwimmingPool", "JoggingTrack", "IndoorGames", "SportsFacility", "GolfCourse"],
        "🏢 Building Facilities": ["LiftAvailable", "CarParking", "PowerBackup", "24X7Security", "MaintenanceStaff", "Intercom"],
        "🌿 Lifestyle": ["LandscapedGardens", "ClubHouse", "Cafeteria", "MultipurposeRoom", "Children'splayarea"],
        "🏪 Nearby": ["ShoppingMall", "ATM", "School", "Hospital"],
        "🏠 Home Appliances": ["WashingMachine", "AC", "Wifi", "Microwave", "TV", "DiningTable", "Sofa", "Wardrobe", "Refrigerator", "Gasconnection"],
        "✨ Other": ["StaffQuarter", "RainWaterHarvesting", "BED", "VaastuCompliant"],
    }

    selected_amenities = set()
    for group_name, group_amenities in amenity_groups.items():
        valid = [a for a in group_amenities if a in amenity_cols]
        if valid:
            with st.expander(group_name, expanded=False):
                cols = st.columns(2)
                for i, amenity in enumerate(valid):
                    display = amenity.replace("'", "").replace("Children", "Kids")
                    if cols[i % 2].checkbox(display, key=f"amenity_{amenity}"):
                        selected_amenities.add(amenity)

# ─────────────────────────────────────────────────────────────────────────────
# Predict Button
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
predict_col, _ = st.columns([1, 2])
with predict_col:
    predict_btn = st.button("🔮 Predict House Price", use_container_width=True)

if predict_btn:
    with st.spinner("Analyzing property data..."):
        time.sleep(0.5)

        # Build feature vector
        amenity_values = {col: (1 if col in selected_amenities else 0) for col in amenity_cols}
        amenity_score  = sum(amenity_values.values())

        # Encode city
        try:
            city_enc = city_encoder.transform([city])[0]
        except Exception:
            city_enc = 0

        # Encode location
        loc_clean = location_input if location_input in top_locations else "Other"
        try:
            loc_enc = loc_encoder.transform([loc_clean])[0]
        except Exception:
            loc_enc = 0

        # Build input row
        input_dict = {
            "Area":            sqft,
            "No. of Bedrooms": bedrooms,
            "Resale":          resale_val,
            "amenity_score":   amenity_score,
            "City_encoded":    city_enc,
            "Location_encoded": loc_enc,
        }
        input_dict.update(amenity_values)

        input_df = pd.DataFrame([input_dict])[features]

        # Predict
        log_pred = model.predict(input_df)[0]
        predicted_price = np.expm1(log_pred)
        price_per_sqft  = predicted_price / sqft

    # ── Results ──────────────────────────────────────────────────────────────
    def fmt_price(p):
        if p >= 1e7:
            return f"₹{p/1e7:.2f} Cr"
        elif p >= 1e5:
            return f"₹{p/1e5:.2f} L"
        else:
            return f"₹{p:,.0f}"

    st.markdown(f"""
    <div class="prediction-card">
        <h2>Estimated Property Value</h2>
        <div class="price">{fmt_price(predicted_price)}</div>
        <div class="sub">📍 {location_input}, {city} &nbsp;|&nbsp; 🏠 {sqft} sq ft &nbsp;|&nbsp; 🛏️ {bedrooms} BHK</div>
    </div>
    """, unsafe_allow_html=True)

    # Metrics row
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(f"""<div class="metric-card">
            <div class="label">Price / Sq Ft</div>
            <div class="value">₹{price_per_sqft:,.0f}</div>
        </div>""", unsafe_allow_html=True)
    with m2:
        st.markdown(f"""<div class="metric-card">
            <div class="label">Amenities</div>
            <div class="value">{amenity_score} / {len(amenity_cols)}</div>
        </div>""", unsafe_allow_html=True)
    with m3:
        st.markdown(f"""<div class="metric-card">
            <div class="label">Model Used</div>
            <div class="value" style="font-size:0.9rem">{model_name.split()[0]}</div>
        </div>""", unsafe_allow_html=True)
    with m4:
        r2 = results[model_name]["R2"]
        st.markdown(f"""<div class="metric-card">
            <div class="label">Model R²</div>
            <div class="value">{r2:.3f}</div>
        </div>""", unsafe_allow_html=True)

    # Price range estimate (±15%)
    low  = predicted_price * 0.85
    high = predicted_price * 1.15
    st.markdown(f"""
    <div class="info-box">
        💡 <strong>Estimated Price Range:</strong> {fmt_price(low)} – {fmt_price(high)}
        &nbsp;(±15% confidence interval based on market variability)
    </div>
    """, unsafe_allow_html=True)

    # Map
    if show_map:
        with st.spinner("Fetching location coordinates..."):
            geo = geocode_location(city, location_input if location_input != "Other" else "")
        if geo["success"]:
            st.markdown("### 📍 Property Location")
            map_df = pd.DataFrame({"lat": [geo["lat"]], "lon": [geo["lon"]]})
            st.map(map_df, zoom=12)
            st.caption(f"📌 {geo['address']}")
        else:
            st.info("📍 Map unavailable — geocoding service could not resolve this location.")

# ─────────────────────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#4a5568; font-size:0.85rem; padding:1rem 0;">
    🏠 Indian Smart House Price Predictor &nbsp;|&nbsp;
    Built with Streamlit &amp; Scikit-learn &nbsp;|&nbsp;
    Data: 6 Indian Cities (~33K listings)
</div>
""", unsafe_allow_html=True)
