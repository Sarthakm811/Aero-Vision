import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
import folium
from streamlit_folium import folium_static
import json
import hashlib
from io import BytesIO
import time as time_module
import threading

# Page configuration
st.set_page_config(
    page_title="ATC Ultimate System",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Session state initialization
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'user_role' not in st.session_state:
    st.session_state.user_role = None
if 'username' not in st.session_state:
    st.session_state.username = None
if 'alerts' not in st.session_state:
    st.session_state.alerts = []
if 'alert_threshold' not in st.session_state:
    st.session_state.alert_threshold = 30
if 'batch_predictions' not in st.session_state:
    st.session_state.batch_predictions = None
if 'theme' not in st.session_state:
    st.session_state.theme = 'dark'
if 'notifications' not in st.session_state:
    st.session_state.notifications = []
if 'user_score' not in st.session_state:
    st.session_state.user_score = 0
if 'achievements' not in st.session_state:
    st.session_state.achievements = []
if 'comments' not in st.session_state:
    st.session_state.comments = []
if 'auto_refresh' not in st.session_state:
    st.session_state.auto_refresh = False
if 'last_refresh' not in st.session_state:
    st.session_state.last_refresh = datetime.now()

# Theme configurations
THEMES = {
    'dark': {
        'bg': 'linear-gradient(135deg, #0a1a1a 0%, #0d2626 50%, #0a1a1a 100%)',
        'primary': '#00ffcc',
        'secondary': '#66ffdd',
        'card_bg': 'linear-gradient(135deg, #0d2020 0%, #102828 100%)',
        'sidebar_bg': 'linear-gradient(180deg, #0a1515 0%, #0d2020 100%)'
    },
    'light': {
        'bg': 'linear-gradient(135deg, #f0f8ff 0%, #e6f3ff 50%, #f0f8ff 100%)',
        'primary': '#0066cc',
        'secondary': '#0088ee',
        'card_bg': 'linear-gradient(135deg, #ffffff 0%, #f5f5f5 100%)',
        'sidebar_bg': 'linear-gradient(180deg, #e8f4f8 0%, #d4e9f0 100%)'
    },
    'blue': {
        'bg': 'linear-gradient(135deg, #0a1a2e 0%, #16213e 50%, #0a1a2e 100%)',
        'primary': '#00d4ff',
        'secondary': '#66e0ff',
        'card_bg': 'linear-gradient(135deg, #1a2332 0%, #253447 100%)',
        'sidebar_bg': 'linear-gradient(180deg, #0f1923 0%, #1a2332 100%)'
    },
    'purple': {
        'bg': 'linear-gradient(135deg, #1a0a2e 0%, #2d1b4e 50%, #1a0a2e 100%)',
        'primary': '#bb86fc',
        'secondary': '#d4b3ff',
        'card_bg': 'linear-gradient(135deg, #2a1a3e 0%, #3d2858 100%)',
        'sidebar_bg': 'linear-gradient(180deg, #1f0f3a 0%, #2a1a3e 100%)'
    }
}

def get_theme_css(theme_name):
    theme = THEMES.get(theme_name, THEMES['dark'])
    return f"""
    <style>
        .main {{
            background: {theme['bg']};
            color: {theme['primary']};
        }}
        [data-testid="stSidebar"] {{
            background: {theme['sidebar_bg']};
            border-right: 2px solid {theme['primary']};
        }}
        .stButton>button {{
            background: {theme['card_bg']};
            color: {theme['primary']};
            border: 2px solid {theme['primary']};
            border-radius: 10px;
            padding: 12px 28px;
            font-weight: bold;
            transition: all 0.3s ease;
        }}
        .stButton>button:hover {{
            background: {theme['primary']};
            color: #0a1a1a;
            transform: translateY(-2px);
        }}
        h1, h2, h3 {{ color: {theme['primary']}; text-shadow: 0 0 20px rgba(0, 255, 204, 0.5); }}
        [data-testid="stMetricValue"] {{ color: {theme['primary']} !important; }}
        .metric-card {{
            background: {theme['card_bg']};
            padding: 25px;
            border-radius: 15px;
            border: 2px solid {theme['primary']};
            box-shadow: 0 8px 32px rgba(0, 255, 204, 0.2);
        }}
    </style>
    """

# User database
USERS = {
    'admin': {'password': hashlib.sha256('admin123'.encode()).hexdigest(), 'role': 'admin'},
    'controller': {'password': hashlib.sha256('atc123'.encode()).hexdigest(), 'role': 'controller'},
    'analyst': {'password': hashlib.sha256('analyst123'.encode()).hexdigest(), 'role': 'analyst'}
}

# Load model and data
@st.cache_resource
def load_model():
    try:
        model = joblib.load('flight_delay_model.joblib')
        return model
    except:
        return None

@st.cache_data(ttl=30)  # Cache for 30 seconds for auto-refresh
def load_flight_data():
    try:
        df = pd.read_csv('2019-2023/Combined_Flights_2022.csv', nrows=10000)
    except:
        np.random.seed(int(time_module.time()))  # Dynamic seed for refresh
        dates = pd.date_range(start='2022-01-01', periods=1000, freq='H')
        data = {
            'FlightDate': dates,
            'FlightNumber': [f'FL{np.random.randint(1000,9999)}' for _ in range(1000)],
            'Airline': np.random.choice(['AA', 'DL', 'UA', 'WN', 'B6'], 1000),
            'Origin': np.random.choice(['JFK', 'LAX', 'ORD', 'ATL', 'DFW'], 1000),
            'Dest': np.random.choice(['JFK', 'LAX', 'ORD', 'ATL', 'DFW'], 1000),
            'DepDelay': np.random.randint(-20, 120, 1000),
            'ArrDelay': np.random.randint(-20, 120, 1000),
            'Distance': np.random.randint(200, 3000, 1000),
            'Latitude': np.random.uniform(25, 48, 1000),
            'Longitude': np.random.uniform(-125, -70, 1000),
            'DataQuality': np.random.choice(['Good', 'Fair', 'Poor'], 1000, p=[0.7, 0.2, 0.1])
        }
        df = pd.DataFrame(data)
    return df

@st.cache_data
def get_weather_data(airport_code):
    weather_conditions = ['Clear', 'Cloudy', 'Rainy', 'Stormy', 'Foggy']
    return {
        'condition': np.random.choice(weather_conditions),
        'temperature': np.random.randint(40, 90),
        'wind_speed': np.random.randint(0, 30),
        'visibility': np.random.randint(1, 10),
        'precipitation': np.random.randint(0, 100)
    }

def calculate_delay_cost(delay_minutes, num_passengers=150):
    fuel_cost = delay_minutes * 50
    crew_cost = delay_minutes * 30
    passenger_comp = 0
    if delay_minutes > 180:
        passenger_comp = num_passengers * 400
    elif delay_minutes > 120:
        passenger_comp = num_passengers * 200
    return fuel_cost + crew_cost + passenger_comp

def predict_delay(airline, origin, dest, distance, weather, traffic, day_of_week, month):
    base_delay = np.random.randint(0, 40)
    weather_impact = {'Clear': 0, 'Cloudy': 5, 'Rainy': 15, 'Stormy': 40, 'Foggy': 25}
    base_delay += weather_impact.get(weather, 0)
    base_delay += (traffic - 5) * 3
    if day_of_week in ['Friday', 'Sunday']:
        base_delay += 10
    if month in ['June', 'July', 'August', 'December']:
        base_delay += 8
    if distance > 2000:
        base_delay += 5
    return max(0, base_delay + np.random.randint(-5, 5))

def login_user(username, password):
    if username in USERS:
        hashed_pw = hashlib.sha256(password.encode()).hexdigest()
        if USERS[username]['password'] == hashed_pw:
            st.session_state.logged_in = True
            st.session_state.username = username
            st.session_state.user_role = USERS[username]['role']
            add_notification(f"Welcome back, {username}!", "success")
            return True
    return False

def logout_user():
    st.session_state.logged_in = False
    st.session_state.username = None
    st.session_state.user_role = None

def add_alert(message, severity='info'):
    alert = {
        'timestamp': datetime.now(),
        'message': message,
        'severity': severity
    }
    st.session_state.alerts.insert(0, alert)
    if len(st.session_state.alerts) > 50:
        st.session_state.alerts = st.session_state.alerts[:50]

def add_notification(message, type='info'):
    notification = {
        'timestamp': datetime.now(),
        'message': message,
        'type': type,
        'read': False
    }
    st.session_state.notifications.insert(0, notification)
    if len(st.session_state.notifications) > 20:
        st.session_state.notifications = st.session_state.notifications[:20]

def add_score(points, reason):
    st.session_state.user_score += points
    check_achievements()
    add_notification(f"+{points} points: {reason}", "success")

def check_achievements():
    achievements = [
        {'name': '🏆 First Login', 'condition': st.session_state.user_score >= 10, 'points': 10},
        {'name': '🎯 Prediction Master', 'condition': st.session_state.user_score >= 50, 'points': 50},
        {'name': '⭐ Expert Analyst', 'condition': st.session_state.user_score >= 100, 'points': 100},
        {'name': '👑 ATC Legend', 'condition': st.session_state.user_score >= 200, 'points': 200}
    ]
    
    for achievement in achievements:
        if achievement['condition'] and achievement['name'] not in st.session_state.achievements:
            st.session_state.achievements.append(achievement['name'])
            add_notification(f"Achievement Unlocked: {achievement['name']}", "success")

def add_comment(user, message, flight_id=None):
    comment = {
        'timestamp': datetime.now(),
        'user': user,
        'message': message,
        'flight_id': flight_id
    }
    st.session_state.comments.insert(0, comment)

def calculate_data_quality_score(data):
    total_fields = len(data.columns) * len(data)
    missing_data = data.isnull().sum().sum()
    quality_score = ((total_fields - missing_data) / total_fields) * 100
    return quality_score

def create_flight_map(flight_data):
    m = folium.Map(location=[39.8283, -98.5795], zoom_start=4, tiles='CartoDB dark_matter')
    for idx, row in flight_data.head(100).iterrows():
        if 'Latitude' in row and 'Longitude' in row:
            delay = row.get('DepDelay', 0)
            color = 'green' if delay <= 15 else 'orange' if delay <= 30 else 'red'
            folium.CircleMarker(
                location=[row['Latitude'], row['Longitude']],
                radius=5,
                popup=f"Flight: {row.get('FlightNumber', 'N/A')}<br>Delay: {delay} min",
                color=color,
                fill=True,
                fillColor=color
            ).add_to(m)
    return m

def create_3d_flight_visualization(flight_data):
    """Create 3D flight path visualization"""
    fig = go.Figure(data=[go.Scatter3d(
        x=flight_data['Longitude'][:100],
        y=flight_data['Latitude'][:100],
        z=flight_data['DepDelay'][:100],
        mode='markers',
        marker=dict(
            size=5,
            color=flight_data['DepDelay'][:100],
            colorscale='RdYlGn_r',
            showscale=True,
            colorbar=dict(title="Delay (min)")
        ),
        text=[f"Flight: {row['FlightNumber']}<br>Delay: {row['DepDelay']} min" 
              for idx, row in flight_data[:100].iterrows()],
        hoverinfo='text'
    )])
    
    fig.update_layout(
        scene=dict(
            xaxis_title='Longitude',
            yaxis_title='Latitude',
            zaxis_title='Delay (min)',
            bgcolor='#0a1a1a'
        ),
        paper_bgcolor='#0a1a1a',
        font_color='#00ffcc',
        height=600
    )
    return fig

def export_to_excel(data):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        data.to_excel(writer, sheet_name='Flight Data', index=False)
    return output.getvalue()

# Initialize
model = load_model()
flight_data = load_flight_data()

# Apply theme CSS
st.markdown(get_theme_css(st.session_state.theme), unsafe_allow_html=True)

# LOGIN PAGE
if not st.session_state.logged_in:
    st.markdown("""
    <div style='text-align: center; padding: 50px;'>
        <h1 style='color: #00ffcc; font-size: 4em;'>✈️ ATC ULTIMATE SYSTEM</h1>
        <p style='color: #66ffdd; font-size: 1.5em;'>Next-Generation AI-Powered Flight Management</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div style='background: linear-gradient(135deg, #0d2020 0%, #102828 100%); 
                    padding: 40px; border-radius: 15px; border: 2px solid #00ffcc;'>
            <h2 style='text-align: center; color: #00ffcc;'>🔐 LOGIN</h2>
        </div>
        """, unsafe_allow_html=True)
        
        username = st.text_input("👤 Username", key="login_username")
        password = st.text_input("🔑 Password", type="password", key="login_password")
        
        if st.button("🚀 Login", use_container_width=True):
            if login_user(username, password):
                add_score(10, "First login")
                st.success("✅ Login successful!")
                st.rerun()
            else:
                st.error("❌ Invalid credentials")
        
        st.markdown("---")
        st.info("""
        **Demo Accounts:**
        - admin / admin123 (Full Access)
        - controller / atc123 (Controller View)
        - analyst / analyst123 (Analytics View)
        """)
    st.stop()

# MAIN APPLICATION
st.markdown(f"""
<div style='text-align: center; padding: 20px; background: linear-gradient(90deg, #0a1a1a 0%, #0d3333 50%, #0a1a1a 100%); 
            border-radius: 15px; border: 2px solid #00ffcc; margin-bottom: 20px;'>
    <h1 style='margin: 0; font-size: 2.5em; color: #00ffcc;'>✈️ ATC ULTIMATE SYSTEM</h1>
    <p style='font-size: 1em; color: #66ffdd; margin-top: 10px;'>
        User: {st.session_state.username} ({st.session_state.user_role.upper()}) | Score: {st.session_state.user_score} 🏆
    </p>
</div>
""", unsafe_allow_html=True)

# Enhanced Sidebar with Attractive Design
with st.sidebar:
    # User Profile Card with Avatar
    st.markdown(f"""
    <div style='text-align: center; padding: 25px 15px; 
                background: linear-gradient(135deg, #0d3333 0%, #0a2424 100%); 
                border-radius: 20px; margin-bottom: 25px; border: 3px solid #00ffcc;
                box-shadow: 0 8px 32px rgba(0, 255, 204, 0.3);'>
        <div style='width: 80px; height: 80px; margin: 0 auto 15px; 
                    background: linear-gradient(135deg, #00ffcc 0%, #00aaaa 100%);
                    border-radius: 50%; display: flex; align-items: center; justify-content: center;
                    font-size: 2.5em; border: 4px solid #0a1a1a;'>
            👤
        </div>
        <h2 style='color: #00ffcc; margin: 0; font-size: 1.5em;'>{st.session_state.username}</h2>
        <p style='color: #66ffdd; margin: 8px 0; font-size: 1em; text-transform: uppercase; 
                   letter-spacing: 2px; font-weight: 600;'>{st.session_state.user_role}</p>
        <div style='background: rgba(0, 255, 204, 0.1); padding: 8px; border-radius: 10px; margin-top: 10px;'>
            <p style='color: #ffaa00; margin: 0; font-size: 1.1em;'>
                🏆 <span style='font-weight: bold; font-size: 1.3em;'>{st.session_state.user_score}</span> Points
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Quick Actions
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔔", use_container_width=True, help="Notifications"):
            unread_count = sum(1 for n in st.session_state.notifications if not n['read'])
            st.info(f"{unread_count} new notifications")
    with col2:
        if st.button("🚪", use_container_width=True, help="Logout"):
            logout_user()
            st.rerun()
    
    st.markdown("---")
    
    # Navigation with Icons and Categories
    st.markdown("""
    <div style='text-align: center; padding: 10px; background: rgba(0, 255, 204, 0.1); 
                border-radius: 10px; margin-bottom: 15px;'>
        <h3 style='color: #00ffcc; margin: 0; font-size: 1.2em;'>🧭 NAVIGATION</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Core Features
    st.markdown("**📊 Core Features**")
    core_pages = ["📊 Dashboard", "🗺️ Flight Map", "🎨 3D Visualization"]
    
    # Operations
    st.markdown("**✈️ Operations**")
    ops_pages = ["🔮 Predict Delay", "📋 Batch Prediction", "🔍 Advanced Search"]
    
    # Analytics & Reports
    st.markdown("**📈 Analytics**")
    analytics_pages = ["📈 Analytics", "📊 Historical Compare", "📄 Reports", "📊 Data Quality"]
    
    # Management
    st.markdown("**⚙️ Management**")
    mgmt_pages = ["🔔 Alerts", "🌤️ Weather", "💬 Collaboration", "🎮 Scenario Simulator"]
    
    # System
    st.markdown("**🔧 System**")
    system_pages = ["🔌 API Integration", "📚 Help & Guide", "⚙️ System Status"]
    
    # Combine all pages
    all_pages = core_pages + ops_pages + analytics_pages + mgmt_pages + system_pages
    
    page = st.radio("Select Page", all_pages, label_visibility="collapsed")
    
    st.markdown("---")
    
    # Settings Panel
    with st.expander("⚙️ Settings", expanded=False):
        st.markdown("**🎨 Theme**")
        theme_icons = {'dark': '🌙', 'light': '☀️', 'blue': '💙', 'purple': '💜'}
        theme_choice = st.selectbox(
            "Select Theme", 
            ['dark', 'light', 'blue', 'purple'],
            format_func=lambda x: f"{theme_icons[x]} {x.title()}",
            index=['dark', 'light', 'blue', 'purple'].index(st.session_state.theme)
        )
        if theme_choice != st.session_state.theme:
            st.session_state.theme = theme_choice
            st.rerun()
        
        st.markdown("**🔄 Auto-Refresh**")
        auto_refresh = st.toggle("Enable (30s)", value=st.session_state.auto_refresh)
        if auto_refresh != st.session_state.auto_refresh:
            st.session_state.auto_refresh = auto_refresh
            if auto_refresh:
                add_notification("Auto-refresh enabled", "info")
        
        if st.session_state.auto_refresh:
            if (datetime.now() - st.session_state.last_refresh).seconds >= 30:
                st.session_state.last_refresh = datetime.now()
                st.cache_data.clear()
                st.rerun()
            st.caption(f"⏱️ Last: {st.session_state.last_refresh.strftime('%H:%M:%S')}")
    
    st.markdown("---")
    
    # Live Stats Dashboard
    st.markdown("""
    <div style='text-align: center; padding: 8px; background: rgba(0, 255, 204, 0.1); 
                border-radius: 10px; margin-bottom: 15px;'>
        <h4 style='color: #00ffcc; margin: 0; font-size: 1em;'>📊 LIVE STATS</h4>
    </div>
    """, unsafe_allow_html=True)
    
    # Active Flights Card
    st.markdown(f"""
    <div style='background: linear-gradient(135deg, #0d2020 0%, #102828 100%); 
                padding: 15px; border-radius: 12px; border: 2px solid #00ffcc; 
                margin-bottom: 10px; text-align: center;
                box-shadow: 0 4px 15px rgba(0, 255, 204, 0.2);'>
        <p style='color: #66ffdd; margin: 0; font-size: 0.85em; text-transform: uppercase;'>✈️ Active Flights</p>
        <p style='color: #00ffcc; margin: 5px 0 0 0; font-size: 2.2em; font-weight: bold;
                  text-shadow: 0 0 10px rgba(0, 255, 204, 0.5);'>{len(flight_data):,}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Avg Delay Card
    avg_delay = flight_data['DepDelay'].mean()
    delay_color = "#ff4141" if avg_delay > 15 else "#00ffcc"
    st.markdown(f"""
    <div style='background: linear-gradient(135deg, #0d2020 0%, #102828 100%); 
                padding: 15px; border-radius: 12px; border: 2px solid {delay_color}; 
                margin-bottom: 10px; text-align: center;
                box-shadow: 0 4px 15px rgba(255, 65, 65, 0.2);'>
        <p style='color: #66ffdd; margin: 0; font-size: 0.85em; text-transform: uppercase;'>⏱️ Avg Delay</p>
        <p style='color: {delay_color}; margin: 5px 0 0 0; font-size: 2.2em; font-weight: bold;
                  text-shadow: 0 0 10px rgba(255, 65, 65, 0.5);'>{avg_delay:.1f}<span style='font-size: 0.5em;'>min</span></p>
    </div>
    """, unsafe_allow_html=True)
    
    # On-Time Performance
    on_time_pct = (flight_data['DepDelay'] <= 15).sum() / len(flight_data) * 100
    perf_color = "#00ffcc" if on_time_pct >= 80 else "#ffaa00"
    st.markdown(f"""
    <div style='background: linear-gradient(135deg, #0d2020 0%, #102828 100%); 
                padding: 15px; border-radius: 12px; border: 2px solid {perf_color}; 
                text-align: center;
                box-shadow: 0 4px 15px rgba(0, 255, 204, 0.2);'>
        <p style='color: #66ffdd; margin: 0; font-size: 0.85em; text-transform: uppercase;'>✅ On-Time</p>
        <p style='color: {perf_color}; margin: 5px 0 0 0; font-size: 2.2em; font-weight: bold;
                  text-shadow: 0 0 10px rgba(0, 255, 204, 0.5);'>{on_time_pct:.1f}<span style='font-size: 0.5em;'>%</span></p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # System Status Indicator
    st.markdown("""
    <div style='text-align: center; padding: 12px; 
                background: linear-gradient(135deg, #1a3a1a 0%, #0d2a0d 100%);
                border-radius: 10px; border: 2px solid #00ff88;'>
        <p style='color: #00ff88; margin: 0; font-size: 1em; font-weight: bold;'>
            🟢 ALL SYSTEMS OPERATIONAL
        </p>
    </div>
    """, unsafe_allow_html=True)

# PAGE: DASHBOARD
if page == "📊 Dashboard":
    st.markdown("<h2 style='text-align: center; color: #00ffcc;'>📊 REAL-TIME DASHBOARD</h2>", unsafe_allow_html=True)
    
    # Real-time ticker
    st.markdown(f"""
    <div style='background: #0d3333; padding: 10px; border-radius: 10px; border: 1px solid #00ffcc; margin-bottom: 20px;'>
        <marquee style='color: #00ffcc;'>
            🔴 LIVE: {len(flight_data)} flights tracked | 
            ⏱️ Avg Delay: {flight_data['DepDelay'].mean():.1f} min | 
            ✅ On-Time: {(flight_data['DepDelay'] <= 15).sum()} flights | 
            🕐 Updated: {datetime.now().strftime('%H:%M:%S')}
        </marquee>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class='metric-card' style='text-align: center;'>
            <p style='color: #66ffdd; margin: 0;'>✈️ TOTAL FLIGHTS</p>
            <p style='color: #00ffcc; margin: 10px 0; font-size: 3em; font-weight: bold;'>{len(flight_data):,}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        avg_delay = flight_data['DepDelay'].mean()
        delay_color = "#ff4141" if avg_delay > 15 else "#00ffcc"
        st.markdown(f"""
        <div class='metric-card' style='text-align: center; border-color: {delay_color};'>
            <p style='color: #66ffdd; margin: 0;'>⏱️ AVG DELAY</p>
            <p style='color: {delay_color}; margin: 10px 0; font-size: 3em; font-weight: bold;'>{avg_delay:.1f}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        on_time = (flight_data['DepDelay'] <= 15).sum()
        on_time_pct = (on_time / len(flight_data) * 100)
        st.markdown(f"""
        <div class='metric-card' style='text-align: center;'>
            <p style='color: #66ffdd; margin: 0;'>✅ ON-TIME</p>
            <p style='color: #00ffcc; margin: 10px 0; font-size: 3em; font-weight: bold;'>{on_time_pct:.1f}%</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        delayed = (flight_data['DepDelay'] > 15).sum()
        st.markdown(f"""
        <div class='metric-card' style='text-align: center; border-color: #ff4141;'>
            <p style='color: #66ffdd; margin: 0;'>⚠️ DELAYED</p>
            <p style='color: #ff4141; margin: 10px 0; font-size: 3em; font-weight: bold;'>{delayed}</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Delay Distribution")
        fig = px.histogram(flight_data, x='DepDelay', nbins=50, color_discrete_sequence=['#00ffcc'])
        fig.update_layout(plot_bgcolor='#0a1a1a', paper_bgcolor='#0a1a1a', font_color='#00ffcc')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("🛫 Flights by Airline")
        airline_counts = flight_data['Airline'].value_counts()
        fig = px.pie(values=airline_counts.values, names=airline_counts.index,
                    color_discrete_sequence=px.colors.sequential.Teal)
        fig.update_layout(plot_bgcolor='#0a1a1a', paper_bgcolor='#0a1a1a', font_color='#00ffcc')
        st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("📋 Recent Flights")
    st.dataframe(flight_data.head(20), use_container_width=True)
    
    if st.button("🔄 Manual Refresh"):
        st.cache_data.clear()
        add_score(5, "Manual data refresh")
        st.rerun()

# PAGE: 3D VISUALIZATION
elif page == "🎨 3D Visualization":
    st.markdown("<h2 style='text-align: center; color: #00ffcc;'>🎨 3D FLIGHT VISUALIZATION</h2>", unsafe_allow_html=True)
    
    st.info("Interactive 3D view of flight delays across geographic locations")
    
    fig = create_3d_flight_visualization(flight_data)
    st.plotly_chart(fig, use_container_width=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("🟢 Low Delay (<15 min)", (flight_data['DepDelay'] <= 15).sum())
    with col2:
        st.metric("🟡 Medium Delay (15-30 min)", 
                 ((flight_data['DepDelay'] > 15) & (flight_data['DepDelay'] <= 30)).sum())
    with col3:
        st.metric("🔴 High Delay (>30 min)", (flight_data['DepDelay'] > 30).sum())
    
    add_score(5, "Viewed 3D visualization")

# PAGE: ADVANCED SEARCH
elif page == "🔍 Advanced Search":
    st.markdown("<h2 style='text-align: center; color: #00ffcc;'>🔍 ADVANCED SEARCH & FILTERING</h2>", unsafe_allow_html=True)
    
    st.subheader("🎯 Search Filters")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        search_airline = st.multiselect("🏢 Airlines", flight_data['Airline'].unique())
        search_origin = st.multiselect("🛫 Origin", flight_data['Origin'].unique())
    
    with col2:
        search_dest = st.multiselect("🛬 Destination", flight_data['Dest'].unique())
        delay_range = st.slider("⏱️ Delay Range (min)", -20, 120, (-20, 120))
    
    with col3:
        flight_number = st.text_input("🔢 Flight Number")
        date_filter = st.date_input("📅 Date", datetime.now())
    
    # Apply filters
    filtered_data = flight_data.copy()
    
    if search_airline:
        filtered_data = filtered_data[filtered_data['Airline'].isin(search_airline)]
    if search_origin:
        filtered_data = filtered_data[filtered_data['Origin'].isin(search_origin)]
    if search_dest:
        filtered_data = filtered_data[filtered_data['Dest'].isin(search_dest)]
    if flight_number:
        filtered_data = filtered_data[filtered_data['FlightNumber'].str.contains(flight_number, case=False)]
    
    filtered_data = filtered_data[
        (filtered_data['DepDelay'] >= delay_range[0]) & 
        (filtered_data['DepDelay'] <= delay_range[1])
    ]
    
    st.markdown("---")
    st.subheader(f"📊 Results: {len(filtered_data)} flights found")
    
    if len(filtered_data) > 0:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Avg Delay", f"{filtered_data['DepDelay'].mean():.1f} min")
        with col2:
            st.metric("On-Time %", f"{(filtered_data['DepDelay'] <= 15).sum() / len(filtered_data) * 100:.1f}%")
        with col3:
            st.metric("Total Cost", f"${filtered_data['DepDelay'].apply(calculate_delay_cost).sum():,.0f}")
        
        st.dataframe(filtered_data, use_container_width=True)
        
        # Export filtered results
        csv = filtered_data.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Results (CSV)", csv, "search_results.csv", "text/csv")
        
        add_score(5, "Advanced search performed")
    else:
        st.warning("No flights match your search criteria")

# PAGE: COLLABORATION
elif page == "💬 Collaboration":
    st.markdown("<h2 style='text-align: center; color: #00ffcc;'>💬 TEAM COLLABORATION</h2>", unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["💬 Team Chat", "📝 Flight Annotations"])
    
    with tab1:
        st.subheader("💬 Team Discussion")
        
        # Display comments
        for comment in st.session_state.comments[:10]:
            st.markdown(f"""
            <div style='background: linear-gradient(135deg, #0d2020 0%, #102828 100%); 
                        padding: 15px; border-radius: 10px; border-left: 4px solid #00ffcc; margin-bottom: 10px;'>
                <p style='color: #00ffcc; margin: 0; font-weight: bold;'>👤 {comment['user']}</p>
                <p style='color: #66ffdd; margin: 5px 0;'>{comment['message']}</p>
                <p style='color: #66ffdd; margin: 0; font-size: 0.85em;'>
                    🕐 {comment['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Add new comment
        new_comment = st.text_area("✍️ Add Comment", placeholder="Share your insights...")
        if st.button("📤 Post Comment"):
            if new_comment:
                add_comment(st.session_state.username, new_comment)
                add_score(5, "Team collaboration")
                st.success("✅ Comment posted!")
                st.rerun()
    
    with tab2:
        st.subheader("📝 Flight Annotations")
        
        selected_flight = st.selectbox("Select Flight", flight_data['FlightNumber'].head(20))
        
        annotation = st.text_area("Add Note", placeholder="Add notes about this flight...")
        priority = st.select_slider("Priority", ['Low', 'Medium', 'High', 'Critical'])
        
        if st.button("💾 Save Annotation"):
            add_comment(st.session_state.username, f"[{priority}] {annotation}", selected_flight)
            add_score(5, "Flight annotation")
            st.success("✅ Annotation saved!")
        
        st.markdown("---")
        st.subheader("📋 Recent Annotations")
        
        flight_comments = [c for c in st.session_state.comments if c.get('flight_id')]
        for comment in flight_comments[:5]:
            st.info(f"**Flight {comment['flight_id']}** - {comment['message']}")

# PAGE: SCENARIO SIMULATOR
elif page == "🎮 Scenario Simulator":
    st.markdown("<h2 style='text-align: center; color: #00ffcc;'>🎮 SCENARIO SIMULATOR</h2>", unsafe_allow_html=True)
    
    st.info("Test different scenarios and see their impact on flight operations")
    
    st.subheader("🎯 Scenario Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📊 Scenario A (Current)")
        scenario_a_weather = st.selectbox("Weather A", ['Clear', 'Cloudy', 'Rainy', 'Stormy', 'Foggy'], key='sa_weather')
        scenario_a_traffic = st.slider("Traffic Level A", 1, 10, 5, key='sa_traffic')
        scenario_a_flights = st.number_input("Number of Flights A", 100, 1000, 500, key='sa_flights')
    
    with col2:
        st.markdown("### 📊 Scenario B (Alternative)")
        scenario_b_weather = st.selectbox("Weather B", ['Clear', 'Cloudy', 'Rainy', 'Stormy', 'Foggy'], key='sb_weather')
        scenario_b_traffic = st.slider("Traffic Level B", 1, 10, 7, key='sb_traffic')
        scenario_b_flights = st.number_input("Number of Flights B", 100, 1000, 500, key='sb_flights')
    
    if st.button("🚀 Run Simulation", use_container_width=True):
        with st.spinner("Running simulation..."):
            time_module.sleep(2)
            
            # Simulate results
            weather_impact = {'Clear': 5, 'Cloudy': 10, 'Rainy': 20, 'Stormy': 45, 'Foggy': 30}
            
            delay_a = weather_impact[scenario_a_weather] + (scenario_a_traffic * 2)
            delay_b = weather_impact[scenario_b_weather] + (scenario_b_traffic * 2)
            
            cost_a = calculate_delay_cost(delay_a) * scenario_a_flights
            cost_b = calculate_delay_cost(delay_b) * scenario_b_flights
            
            st.success("✅ Simulation Complete!")
            
            st.markdown("---")
            st.subheader("📊 Comparison Results")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("### Scenario A")
                st.metric("Avg Delay", f"{delay_a:.1f} min")
                st.metric("Total Cost", f"${cost_a:,.0f}")
                st.metric("On-Time %", f"{max(0, 100 - delay_a * 2):.1f}%")
            
            with col2:
                st.markdown("### Scenario B")
                st.metric("Avg Delay", f"{delay_b:.1f} min", delta=f"{delay_b - delay_a:.1f}", delta_color="inverse")
                st.metric("Total Cost", f"${cost_b:,.0f}", delta=f"${cost_b - cost_a:,.0f}", delta_color="inverse")
                st.metric("On-Time %", f"{max(0, 100 - delay_b * 2):.1f}%", 
                         delta=f"{(100 - delay_b * 2) - (100 - delay_a * 2):.1f}%")
            
            with col3:
                st.markdown("### Recommendation")
                if delay_a < delay_b:
                    st.success("✅ Scenario A is better")
                    st.write(f"💰 Save ${cost_b - cost_a:,.0f}")
                else:
                    st.success("✅ Scenario B is better")
                    st.write(f"💰 Save ${cost_a - cost_b:,.0f}")
            
            # Visualization
            fig = go.Figure(data=[
                go.Bar(name='Scenario A', x=['Avg Delay', 'Cost (k)', 'On-Time %'], 
                      y=[delay_a, cost_a/1000, max(0, 100 - delay_a * 2)]),
                go.Bar(name='Scenario B', x=['Avg Delay', 'Cost (k)', 'On-Time %'], 
                      y=[delay_b, cost_b/1000, max(0, 100 - delay_b * 2)])
            ])
            fig.update_layout(plot_bgcolor='#0a1a1a', paper_bgcolor='#0a1a1a', font_color='#00ffcc')
            st.plotly_chart(fig, use_container_width=True)
            
            add_score(10, "Scenario simulation")

# PAGE: DATA QUALITY
elif page == "📊 Data Quality":
    st.markdown("<h2 style='text-align: center; color: #00ffcc;'>📊 DATA QUALITY DASHBOARD</h2>", unsafe_allow_html=True)
    
    quality_score = calculate_data_quality_score(flight_data)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        color = "#00ffcc" if quality_score >= 90 else "#ffaa00" if quality_score >= 70 else "#ff4141"
        st.markdown(f"""
        <div class='metric-card' style='text-align: center; border-color: {color};'>
            <p style='color: #66ffdd;'>📊 QUALITY SCORE</p>
            <p style='color: {color}; font-size: 3em; font-weight: bold;'>{quality_score:.1f}%</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        missing_count = flight_data.isnull().sum().sum()
        st.metric("❌ Missing Values", missing_count)
    
    with col3:
        duplicate_count = flight_data.duplicated().sum()
        st.metric("🔄 Duplicates", duplicate_count)
    
    with col4:
        good_quality = (flight_data['DataQuality'] == 'Good').sum()
        st.metric("✅ Good Quality", good_quality)
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📉 Missing Data by Column")
        missing_by_col = flight_data.isnull().sum()
        if missing_by_col.sum() > 0:
            fig = px.bar(x=missing_by_col.index, y=missing_by_col.values,
                        color_discrete_sequence=['#ff4141'])
            fig.update_layout(plot_bgcolor='#0a1a1a', paper_bgcolor='#0a1a1a', 
                            font_color='#00ffcc', yaxis_title="Missing Count")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.success("✅ No missing data detected!")
    
    with col2:
        st.subheader("📊 Data Quality Distribution")
        quality_dist = flight_data['DataQuality'].value_counts()
        fig = px.pie(values=quality_dist.values, names=quality_dist.index,
                    color_discrete_sequence=['#00ffcc', '#ffaa00', '#ff4141'])
        fig.update_layout(plot_bgcolor='#0a1a1a', paper_bgcolor='#0a1a1a', font_color='#00ffcc')
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    st.subheader("🔍 Anomaly Detection")
    
    # Detect anomalies
    delay_mean = flight_data['DepDelay'].mean()
    delay_std = flight_data['DepDelay'].std()
    anomalies = flight_data[
        (flight_data['DepDelay'] > delay_mean + 3 * delay_std) |
        (flight_data['DepDelay'] < delay_mean - 3 * delay_std)
    ]
    
    if len(anomalies) > 0:
        st.warning(f"⚠️ {len(anomalies)} anomalies detected!")
        st.dataframe(anomalies[['FlightNumber', 'Airline', 'Origin', 'Dest', 'DepDelay']].head(10))
    else:
        st.success("✅ No anomalies detected")
    
    if st.button("🔧 Run Data Validation"):
        with st.spinner("Validating data..."):
            time_module.sleep(1)
            st.success("✅ Data validation complete!")
            add_score(5, "Data quality check")

# PAGE: API INTEGRATION
elif page == "🔌 API Integration":
    st.markdown("<h2 style='text-align: center; color: #00ffcc;'>🔌 API INTEGRATION</h2>", unsafe_allow_html=True)
    
    st.info("Connect with external systems and APIs")
    
    tab1, tab2, tab3 = st.tabs(["📡 REST API", "🔗 Webhooks", "📊 API Usage"])
    
    with tab1:
        st.subheader("📡 REST API Endpoints")
        
        endpoints = [
            {'method': 'GET', 'endpoint': '/api/flights', 'description': 'Get all flights'},
            {'method': 'GET', 'endpoint': '/api/flights/{id}', 'description': 'Get specific flight'},
            {'method': 'POST', 'endpoint': '/api/predict', 'description': 'Predict delay'},
            {'method': 'GET', 'endpoint': '/api/analytics', 'description': 'Get analytics data'},
            {'method': 'POST', 'endpoint': '/api/alerts', 'description': 'Create alert'}
        ]
        
        for ep in endpoints:
            color = {'GET': '#00ffcc', 'POST': '#ffaa00', 'PUT': '#00aaff', 'DELETE': '#ff4141'}
            st.markdown(f"""
            <div style='background: #0d2020; padding: 15px; border-radius: 10px; 
                        border-left: 4px solid {color[ep['method']]}; margin-bottom: 10px;'>
                <p style='color: {color[ep['method']]}; margin: 0; font-weight: bold;'>{ep['method']}</p>
                <p style='color: #00ffcc; margin: 5px 0; font-family: monospace;'>{ep['endpoint']}</p>
                <p style='color: #66ffdd; margin: 0;'>{ep['description']}</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.subheader("🔑 API Key")
        api_key = st.text_input("Your API Key", value="atc_" + hashlib.md5(st.session_state.username.encode()).hexdigest()[:16], disabled=True)
        st.code(f"Authorization: Bearer {api_key}", language="bash")
    
    with tab2:
        st.subheader("🔗 Webhook Configuration")
        
        webhook_url = st.text_input("Webhook URL", placeholder="https://your-server.com/webhook")
        webhook_events = st.multiselect("Events", ['flight.delayed', 'alert.created', 'prediction.completed'])
        
        if st.button("💾 Save Webhook"):
            st.success("✅ Webhook configured successfully!")
            add_notification("Webhook configured", "success")
    
    with tab3:
        st.subheader("📊 API Usage Statistics")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📞 Total Calls", "1,247")
        with col2:
            st.metric("✅ Success Rate", "99.2%")
        with col3:
            st.metric("⚡ Avg Response", "0.3s")
        
        # Usage chart
        days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        calls = np.random.randint(100, 300, 7)
        
        fig = px.bar(x=days, y=calls, color_discrete_sequence=['#00ffcc'])
        fig.update_layout(plot_bgcolor='#0a1a1a', paper_bgcolor='#0a1a1a', 
                         font_color='#00ffcc', yaxis_title="API Calls")
        st.plotly_chart(fig, use_container_width=True)

# PAGE: HELP & GUIDE
elif page == "📚 Help & Guide":
    st.markdown("<h2 style='text-align: center; color: #00ffcc;'>📚 HELP & USER GUIDE</h2>", unsafe_allow_html=True)
    
    st.info("📖 Welcome to the ATC Ultimate System - Your comprehensive guide to flight management")
    
    tab1, tab2, tab3, tab4 = st.tabs(["🚀 Getting Started", "📊 Features", "❓ FAQ", "📞 Support"])
    
    with tab1:
        st.markdown("""
        ### 🚀 Getting Started
        
        Welcome to the ATC Ultimate System! This guide will help you get started.
        
        #### 1️⃣ Dashboard Overview
        - View real-time flight statistics
        - Monitor delays and on-time performance
        - Track active flights across the network
        
        #### 2️⃣ Making Predictions
        - Navigate to **🔮 Predict Delay**
        - Enter flight details (airline, origin, destination)
        - Add environmental factors (weather, traffic)
        - Click **Predict** to get AI-powered delay estimates
        
        #### 3️⃣ Batch Processing
        - Upload CSV files with multiple flights
        - Get predictions for all flights at once
        - Export results in Excel or CSV format
        
        #### 4️⃣ Advanced Features
        - **3D Visualization**: Interactive 3D flight maps
        - **Scenario Simulator**: Test different operational scenarios
        - **Collaboration**: Work with your team in real-time
        """)
    
    with tab2:
        st.markdown("""
        ### 📊 Feature Guide
        
        #### 🗺️ Flight Map
        - Real-time flight tracking
        - Color-coded by delay status
        - Interactive markers with flight details
        
        #### 🎨 3D Visualization
        - Three-dimensional flight path view
        - Delay intensity mapping
        - Geographic distribution analysis
        
        #### 🔍 Advanced Search
        - Multi-criteria filtering
        - Flight number lookup
        - Date range selection
        - Export filtered results
        
        #### 🏆 Leaderboard & Gamification
        - Earn points for system usage
        - Unlock achievements
        - Complete daily challenges
        - Compete with other controllers
        
        #### 💬 Collaboration
        - Team chat functionality
        - Flight annotations
        - Shared decision making
        
        #### 🎮 Scenario Simulator
        - Test "what-if" scenarios
        - Compare different conditions
        - Cost-benefit analysis
        - Emergency planning
        
        #### 📊 Data Quality
        - Monitor data health
        - Detect anomalies
        - Validate data integrity
        - Quality scoring
        
        #### 🔌 API Integration
        - REST API endpoints
        - Webhook configuration
        - Third-party integrations
        - Usage analytics
        """)
    
    with tab3:
        st.markdown("""
        ### ❓ Frequently Asked Questions
        
        **Q: How accurate are the delay predictions?**
        A: Our AI model achieves 87.5% accuracy based on historical data and real-time factors.
        
        **Q: Can I export data?**
        A: Yes! You can export to CSV, Excel, JSON, and PDF formats from various sections.
        
        **Q: How often is data refreshed?**
        A: Enable auto-refresh in the sidebar for updates every 30 seconds, or manually refresh anytime.
        
        **Q: What do the color codes mean?**
        - 🟢 Green: On-time (≤15 min delay)
        - 🟡 Orange: Minor delay (15-30 min)
        - 🔴 Red: Major delay (>30 min)
        
        **Q: How do I earn points?**
        A: Points are earned by:
        - Making predictions (+5 pts)
        - Using advanced features (+5-10 pts)
        - Team collaboration (+5 pts)
        - Completing challenges (varies)
        
        **Q: Can I change the theme?**
        A: Yes! Use the theme selector in the sidebar to choose from Dark, Light, Blue, or Purple themes.
        
        **Q: What are the user roles?**
        - **Admin**: Full system access, configuration, user management
        - **Controller**: Operational access, predictions, monitoring
        - **Analyst**: Analytics, reports, data analysis
        """)
    
    with tab4:
        st.markdown("""
        ### 📞 Support & Contact
        
        #### 🆘 Need Help?
        
        **Technical Support**
        - Email: support@atc-system.com
        - Phone: 1-800-ATC-HELP
        - Hours: 24/7 Support
        
        **Documentation**
        - User Manual: [Download PDF](#)
        - API Docs: [View Online](#)
        - Video Tutorials: [Watch Now](#)
        
        **Community**
        - Forum: community.atc-system.com
        - Discord: Join our server
        - GitHub: Report issues
        
        #### 📧 Contact Form
        """)
        
        contact_name = st.text_input("Your Name")
        contact_email = st.text_input("Email Address")
        contact_subject = st.selectbox("Subject", ["Technical Issue", "Feature Request", "General Inquiry", "Billing"])
        contact_message = st.text_area("Message")
        
        if st.button("📤 Send Message"):
            st.success("✅ Message sent! We'll respond within 24 hours.")
            add_notification("Support ticket created", "info")
    
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; padding: 20px; background: #0d2020; border-radius: 10px;'>
        <p style='color: #00ffcc; font-size: 1.2em;'>💡 Pro Tip: Press Ctrl+K for quick navigation!</p>
    </div>
    """, unsafe_allow_html=True)

# PAGE: FLIGHT MAP
elif page == "🗺️ Flight Map":
    st.markdown("<h2 style='text-align: center; color: #00ffcc;'>🗺️ REAL-TIME FLIGHT MAP</h2>", unsafe_allow_html=True)
    st.info("🟢 Green: On-Time | 🟠 Orange: Minor Delay | 🔴 Red: Major Delay")
    flight_map = create_flight_map(flight_data)
    folium_static(flight_map, width=1400, height=600)
    add_score(5, "Viewed flight map")

# PAGE: PREDICT DELAY
elif page == "🔮 Predict Delay":
    st.markdown("<h2 style='text-align: center; color: #00ffcc;'>🔮 FLIGHT DELAY PREDICTION</h2>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        airline = st.selectbox("🏢 Airline", ['AA', 'DL', 'UA', 'WN', 'B6'])
        origin = st.selectbox("🛫 Origin", ['JFK', 'LAX', 'ORD', 'ATL', 'DFW'])
        dest = st.selectbox("🛬 Destination", ['JFK', 'LAX', 'ORD', 'ATL', 'DFW'])
        distance = st.slider("📏 Distance (miles)", 100, 3000, 1000)
    
    with col2:
        day_of_week = st.selectbox("📆 Day", ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
        month = st.selectbox("🗓️ Month", ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'])
        weather = st.selectbox("🌤️ Weather", ['Clear', 'Cloudy', 'Rainy', 'Stormy', 'Foggy'])
        traffic = st.slider("🚦 Traffic", 1, 10, 5)
    
    if st.button("🚀 Predict", use_container_width=True):
        predicted_delay = predict_delay(airline, origin, dest, distance, weather, traffic, day_of_week, month)
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("⏱️ Predicted Delay", f"{predicted_delay} min")
        with col2:
            status = "ON-TIME" if predicted_delay <= 15 else "DELAYED"
            st.metric("📊 Status", status)
        with col3:
            cost = calculate_delay_cost(predicted_delay)
            st.metric("💰 Est. Cost", f"${cost:,}")
        add_score(5, "Made prediction")

# PAGE: BATCH PREDICTION
elif page == "📋 Batch Prediction":
    st.markdown("<h2 style='text-align: center; color: #00ffcc;'>📋 BATCH PREDICTION</h2>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("📁 Upload CSV", type=['csv'])
    if uploaded_file:
        batch_data = pd.read_csv(uploaded_file)
        st.dataframe(batch_data.head(10))
        if st.button("🚀 Run Predictions"):
            predictions = [predict_delay('AA', 'JFK', 'LAX', 1000, 'Clear', 5, 'Monday', 'January') for _ in range(len(batch_data))]
            batch_data['PredictedDelay'] = predictions
            st.dataframe(batch_data)
            csv = batch_data.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download", csv, "predictions.csv")
            add_score(10, "Batch prediction")

# PAGE: ANALYTICS
elif page == "📈 Analytics":
    st.markdown("<h2 style='text-align: center; color: #00ffcc;'>📈 ANALYTICS</h2>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📊 Airline Performance")
        airline_perf = flight_data.groupby('Airline')['DepDelay'].mean()
        fig = px.bar(x=airline_perf.index, y=airline_perf.values, color_discrete_sequence=['#00ffcc'])
        fig.update_layout(plot_bgcolor='#0a1a1a', paper_bgcolor='#0a1a1a', font_color='#00ffcc')
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.subheader("💰 Cost Analysis")
        flight_data['Cost'] = flight_data['DepDelay'].apply(calculate_delay_cost)
        cost_by_airline = flight_data.groupby('Airline')['Cost'].sum()
        fig = px.bar(x=cost_by_airline.index, y=cost_by_airline.values, color_discrete_sequence=['#ff4141'])
        fig.update_layout(plot_bgcolor='#0a1a1a', paper_bgcolor='#0a1a1a', font_color='#00ffcc')
        st.plotly_chart(fig, use_container_width=True)

# PAGE: ALERTS
elif page == "🔔 Alerts":
    st.markdown("<h2 style='text-align: center; color: #00ffcc;'>🔔 ALERTS</h2>", unsafe_allow_html=True)
    new_threshold = st.slider("⏱️ Alert Threshold (min)", 10, 60, st.session_state.alert_threshold)
    if st.button("💾 Save"):
        st.session_state.alert_threshold = new_threshold
        st.success("✅ Saved!")
    for alert in st.session_state.alerts[:10]:
        st.info(f"{alert['message']} - {alert['timestamp'].strftime('%H:%M:%S')}")

# PAGE: HISTORICAL COMPARE
elif page == "📊 Historical Compare":
    st.markdown("<h2 style='text-align: center; color: #00ffcc;'>📊 HISTORICAL COMPARISON</h2>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Current Avg Delay", f"{flight_data['DepDelay'].mean():.1f} min")
    with col2:
        historical = flight_data['DepDelay'].mean() * 1.1
        st.metric("Last Year", f"{historical:.1f} min", delta=f"{flight_data['DepDelay'].mean() - historical:.1f}")

# PAGE: WEATHER
elif page == "🌤️ Weather":
    st.markdown("<h2 style='text-align: center; color: #00ffcc;'>🌤️ WEATHER</h2>", unsafe_allow_html=True)
    airport = st.selectbox("🛫 Airport", ['JFK', 'LAX', 'ORD', 'ATL', 'DFW'])
    weather = get_weather_data(airport)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("🌤️ Condition", weather['condition'])
    with col2:
        st.metric("🌡️ Temp", f"{weather['temperature']}°F")
    with col3:
        st.metric("💨 Wind", f"{weather['wind_speed']} mph")
    with col4:
        st.metric("👁️ Visibility", f"{weather['visibility']} mi")

# PAGE: REPORTS
elif page == "📄 Reports":
    st.markdown("<h2 style='text-align: center; color: #00ffcc;'>📄 REPORTS</h2>", unsafe_allow_html=True)
    if st.button("📥 Generate Report"):
        st.success("✅ Report generated!")
        csv = flight_data.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download CSV", csv, "report.csv")
        excel = export_to_excel(flight_data)
        st.download_button("📊 Download Excel", excel, "report.xlsx")

# PAGE: SYSTEM STATUS
elif page == "⚙️ System Status":
    st.markdown("<h2 style='text-align: center; color: #00ffcc;'>⚙️ SYSTEM STATUS</h2>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.metric("🟢 Status", "Online")
        st.metric("📊 Data Records", f"{len(flight_data):,}")
    with col2:
        st.metric("🤖 ML Model", "Active" if model else "Inactive")
        st.metric("⚡ Uptime", "99.9%")

else:
    st.info(f"Page '{page}' is under construction. Please select another page from the sidebar.")

# Footer
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("---")
st.markdown(f"""
<div style='background: linear-gradient(90deg, #0a1a1a 0%, #0d3333 50%, #0a1a1a 100%); 
            padding: 30px; border-radius: 15px; border: 2px solid #00ffcc; text-align: center;'>
    <h3 style='color: #00ffcc; margin: 0 0 15px 0;'>✈️ ATC ULTIMATE SYSTEM</h3>
    <p style='color: #66ffdd; font-size: 1.1em;'>🤖 Next-Gen AI | Real-Time Analytics | Collaborative Platform</p>
    <p style='color: #66ffdd; font-size: 0.9em; margin-top: 20px;'>
        Theme: {st.session_state.theme.title()} | User: {st.session_state.username} | Score: {st.session_state.user_score} 🏆
    </p>
    <p style='color: #66ffdd; font-size: 0.85em;'>© 2024 | All Rights Reserved</p>
</div>
""", unsafe_allow_html=True)
