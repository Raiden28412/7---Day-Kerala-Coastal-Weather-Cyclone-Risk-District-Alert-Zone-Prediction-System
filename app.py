import streamlit as st
import pandas as pd
import requests
import joblib
import numpy as np

# Page Configuration
st.set_page_config(
    page_title="Kerala Fisherfolk Weather & Safety Guide",
    page_icon="🌊",
    layout="centered"
)

# Custom Styling with Coastal Background Image, Gradients, and Animations
st.markdown("""
    <style>
    /* Background Image covering the entire Streamlit app with a subtle dark overlay for readability */
    .stApp {
        background-image: linear-gradient(rgba(15, 23, 42, 0.8), rgba(15, 23, 42, 0.85)), 
                          url('https://images.unsplash.com/photo-1602216056096-3b40cc0c9944?auto=format&fit=crop&w=1920&q=80');
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }

    /* Main Typography */
    .main-title { font-size: 28px; font-weight: 800; color: #FFFFFF; text-align: center; margin-bottom: 0px; text-shadow: 0 2px 4px rgba(0,0,0,0.3); }
    .subtitle { font-size: 15px; color: #E2E8F0; text-align: center; margin-bottom: 25px; text-shadow: 0 1px 2px rgba(0,0,0,0.3); }

    /* Smooth Pulse Animation Keyframes */
    @keyframes pulse-animation {
        0% { transform: scale(1); opacity: 0.95; }
        50% { transform: scale(1.02); opacity: 1; }
        100% { transform: scale(1); opacity: 0.95; }
    }

    /* Interactive Animated Cards */
    .card-safe { 
        background: rgba(209, 250, 229, 0.95); 
        padding: 22px; 
        border-radius: 14px; 
        border-left: 6px solid #10B981; 
        text-align: center; 
        animation: pulse-animation 2.5s infinite ease-in-out;
        box-shadow: 0 8px 20px rgba(16, 185, 129, 0.25);
        backdrop-filter: blur(8px);
    }
    .card-safe h2, .card-safe p { color: #065F46 !important; }

    .card-danger { 
        background: rgba(254, 226, 226, 0.95); 
        padding: 22px; 
        border-radius: 14px; 
        border-left: 6px solid #EF4444; 
        text-align: center; 
        animation: pulse-animation 1.5s infinite ease-in-out;
        box-shadow: 0 8px 20px rgba(239, 68, 68, 0.3);
        backdrop-filter: blur(8px);
    }
    .card-danger h2, .card-danger p { color: #991B1B !important; }

    .card-warning { 
        background: rgba(254, 243, 199, 0.95); 
        padding: 22px; 
        border-radius: 14px; 
        border-left: 6px solid #F59E0B; 
        text-align: center; 
        animation: pulse-animation 2s infinite ease-in-out;
        box-shadow: 0 8px 20px rgba(245, 158, 11, 0.25);
        backdrop-filter: blur(8px);
    }
    .card-warning h2, .card-warning p { color: #92400E !important; }

    /* Make headers and text pop against dark background */
    h2, h3, label { color: #F8FAFC !important; text-shadow: 0 1px 2px rgba(0,0,0,0.5); }
    </style>
""", unsafe_allow_html=True)

# Load Model Safely
@st.cache_resource
def load_model():
    try:
        return joblib.load('kerala_weather_model.pkl')
    except Exception as e:
        return None

model = load_model()

# 9 Coastal Districts of Kerala
district_coords = {
    "Thiruvananthapuram": {"lat": 8.5241, "lon": 76.9366},
    "Kollam": {"lat": 8.8932, "lon": 76.6141},
    "Alappuzha": {"lat": 9.4981, "lon": 76.3388},
    "Ernakulam": {"lat": 9.9816, "lon": 76.2999},
    "Thrissur": {"lat": 10.5276, "lon": 76.2144},
    "Malappuram": {"lat": 11.0734, "lon": 76.0740},
    "Kozhikode": {"lat": 11.2588, "lon": 75.7804},
    "Kannur": {"lat": 11.8745, "lon": 75.3704},
    "Kasaragod": {"lat": 12.4996, "lon": 75.0136}
}

# App Header
st.markdown('<p class="main-title">🌊 കടൽ സുരക്ഷ (Sea Safety & Weather Guide)</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Check if it is safe to go fishing in your district today.</p>', unsafe_allow_html=True)

if model is None:
    st.error("⚠️ Model file `kerala_weather_model.pkl` is missing.")
else:
    # Simple District Selector (Big and Central)
    selected_district = st.selectbox(
        "📍 Select Your District (ജില്ല തിരഞ്ഞെടുക്കുക):", 
        list(district_coords.keys())
    )
    
    lat = district_coords[selected_district]["lat"]
    lon = district_coords[selected_district]["lon"]

    # Fetch Live Data
    @st.cache_data(ttl=3600)
    def fetch_weather(latitude, longitude):
        url = f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&daily=temperature_2m_max,surface_pressure_mean,rain_sum,wind_speed_10m_max,wind_gusts_10m_max&timezone=auto"
        response = requests.get(url)
        if response.status_code == 200:
            return response.json()
        return None

    data = fetch_weather(lat, lon)

    if data and "daily" in data:
        daily = data["daily"]
        
        df_forecast = pd.DataFrame({
            "Date": daily["time"],
            "temperature_2m (°C)": daily["temperature_2m_max"],
            "surface_pressure (hpa)": daily["surface_pressure_mean"],
            "rain (mm)": daily["rain_sum"],
            "wind_speed_10m (km/h)": daily["wind_speed_10m_max"],
            "wind_gusts_10m (km/h)": daily["wind_gusts_10m_max"]
        })
        
        df_forecast.ffill(inplace=True)
        df_forecast.fillna(value=0, inplace=True)
        df_forecast["Flood_Runoff_mm"] = df_forecast["rain (mm)"] * 0.35

        # Convert to NumPy array to cleanly bypass any feature name mismatch errors
        X_live = df_forecast[[
            'surface_pressure (hpa)', 
            'temperature_2m (°C)', 
            'rain (mm)',
            'wind_speed_10m (km/h)', 
            'wind_gusts_10m (km/h)', 
            'Flood_Runoff_mm'
        ]].to_numpy()
        
        # Predict for all 7 days automatically
        df_forecast["Predicted Alert Zone"] = model.predict(X_live)

        st.markdown("---")
        
        # --- TODAY'S STATUS (BIG & CLEAR WITH ANIMATIONS) ---
        st.subheader("Today's Status (ഇന്നത്തെ സ്ഥിതി)")
        today_zone = str(df_forecast['Predicted Alert Zone'][0]).lower()
        today_wind = df_forecast['wind_speed_10m (km/h)'][0]
        today_rain = df_forecast['rain (mm)'][0]

        # Simple metrics for context without jargon
        m1, m2 = st.columns(2)
        m1.metric("🌬️ Wind Speed (കാറ്റിന്റെ വേഗത)", f"{today_wind} km/h")
        m2.metric("🌧️ Rainfall (മഴ)", f"{today_rain} mm")

        st.markdown("<br>", unsafe_allow_html=True)

        if "red" in today_zone:
            st.markdown("""
                <div class="card-danger">
                    <h2>🚨 DO NOT GO TO SEA (കടലിൽ പോകരുത്)</h2>
                    <p>High Danger! Rough sea and storm conditions expected in <b>{}</b> today.</p>
                </div>
            """.format(selected_district), unsafe_allow_html=True)
        elif "orange" in today_zone:
            st.markdown("""
                <div class="card-warning">
                    <h2>⚠️ EXTREME CAUTION (ജാഗ്രത പാലിക്കുക)</h2>
                    <p>Moderate-High Risk. Strong winds and rough waters in <b>{}</b>. Be very careful.</p>
                </div>
            """.format(selected_district), unsafe_allow_html=True)
        elif "yellow" in today_zone:
            st.markdown("""
                <div class="card-warning">
                    <h2>⚡ WEATHER ALERT (മുന്നറിയിപ്പ്)</h2>
                    <p>Minor weather fluctuations expected in <b>{}</b>. Keep watch on updates.</p>
                </div>
            """.format(selected_district), unsafe_allow_html=True)
        else:
            st.markdown("""
                <div class="card-safe">
                    <h2>🟢 SAFE TO FISH (കടലിൽ പോകാം - സുരക്ഷിതം)</h2>
                    <p>Conditions are normal and safe for fishing in <b>{}</b> today.</p>
                </div>
            """.format(selected_district), unsafe_allow_html=True)

        # --- SIMPLE 7-DAY OUTLOOK CARDS ---
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.subheader("📅 Next 7 Days Outlook (വരും ദിവസങ്ങളിലെ അവസ്ഥ)")
        
        for index, row in df_forecast.iterrows():
            date_str = row["Date"]
            zone = row["Predicted Alert Zone"]
            wind = row["wind_speed_10m (km/h)"]
            
            emoji = "🟢 Safe"
            if "red" in str(zone).lower():
                emoji = "🚨 Danger"
            elif "orange" in str(zone).lower() or "yellow" in str(zone).lower():
                emoji = "⚠️ Caution"

            with st.container():
                cols = st.columns([2, 2, 3])
                cols[0].write(f"📅 **{date_str}**")
                cols[1].write(f"Wind: {wind} km/h")
                cols[2].write(f"Status: **{emoji}**")
            st.divider()

    else:
        st.error("⚠️ Failed to load live weather updates.")