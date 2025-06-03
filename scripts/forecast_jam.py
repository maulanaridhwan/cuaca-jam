import os
import json
import time
import datetime
import requests
import pandas as pd
import numpy as np
import pickle
import logging
from tensorflow.keras.models import load_model
from flask import Flask, jsonify
import pytz
import tempfile
from functools import lru_cache
import threading

# Flask app setup
app = Flask(__name__)

# Setup logging untuk Railway
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]  # Hanya console logging untuk Railway
)
logger = logging.getLogger(__name__)

# ================== KONFIGURASI ==================
LATITUDE = -8.65
LONGITUDE = 115.22
WITA = pytz.timezone('Asia/Makassar')

# Features yang akan digunakan untuk prediksi
NUM_FEATS = [
    "temperature_2m",
    "relative_humidity_2m", 
    "precipitation",
    "cloudcover",
    "windspeed_10m",
    "winddirection_10m",
    "surface_pressure",
]

TARGET_FEATURES = [
    "temperature_2m",
    "relative_humidity_2m", 
    "precipitation",
    "cloudcover",
    "windspeed_10m",
    "surface_pressure",
    "winddirection_10m"
]

# API URL untuk Open-Meteo
BASE_API_URL = "https://api.open-meteo.com/v1/forecast"
CURRENT_API_URL = (
    f"{BASE_API_URL}?latitude={LATITUDE}&longitude={LONGITUDE}"
    f"&current=temperature_2m,relative_humidity_2m,precipitation,cloudcover,"
    f"windspeed_10m,winddirection_10m,surface_pressure,weathercode"
)

PAST_HOURS = 24
FUTURE_HOURS = 12

# In-memory cache untuk prediksi (Railway safe)
prediction_cache = {
    'data': None,
    'timestamp': None,
    'lock': threading.Lock()
}

# Global model variables (loaded once)
model_components = {
    'model': None,
    'scaler': None,
    'scaler_precip': None,
    'encoder': None,
    'loaded': False,
    'lock': threading.Lock()
}

# ================== HELPER FUNCTIONS ==================
def deg_to_compass(deg):
    dirs = ['Utara', 'Timur Laut', 'Timur', 'Tenggara',
            'Selatan', 'Barat Daya', 'Barat', 'Barat Laut']
    ix = int((deg + 22.5) // 45) % 8
    return dirs[ix]

def map_weather_group(code):
    """Map weather code ke weather group dengan fallback"""
    try:
        code = int(code)
    except (ValueError, TypeError):
        logger.warning(f"Invalid weather code: {code}, using default 'Cerah'")
        return 'Cerah'
    
    if code in [0, 1, 2]:
        return 'Cerah'
    elif code in [3]:
        return 'Berawan'
    elif code in [51, 53]:
        return 'Gerimis'
    elif code in [45, 48, 55, 56, 57, 61, 63, 65, 66, 67, 71, 73, 75, 77, 80, 81, 82, 85, 86, 95, 96, 99]:
        return 'Hujan'
    else:
        logger.warning(f"Unknown weather code: {code}, mapping to 'Cerah' as fallback")
        return 'Cerah'

def safe_transform_weather_labels(encoder, weather_groups):
    """Safely transform weather groups dengan fallback"""
    try:
        known_classes = set(encoder.classes_)
        unknown_classes = set(weather_groups) - known_classes
        
        if unknown_classes:
            logger.warning(f"Unknown weather groups: {unknown_classes}")
            fallback_class = encoder.classes_[0]
            weather_groups_safe = [
                group if group in known_classes else fallback_class 
                for group in weather_groups
            ]
        else:
            weather_groups_safe = weather_groups
        
        return encoder.transform(weather_groups_safe)
    except Exception as e:
        logger.error(f"Error in safe_transform_weather_labels: {e}")
        return np.zeros(len(weather_groups), dtype=int)

# ================== MODEL LOADING ==================
def find_model_files():
    """Find model files in various locations"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    possible_paths = [
        os.path.join(script_dir, "models"),
        os.path.join(script_dir, "..", "models"),
        script_dir,
        "models",
        "/app/models",  # Railway specific path
        "/opt/render/project/src/models",  # Render specific path
    ]
    
    required_files = {
        "scaler": "hourly_scaler.pkl",
        "scaler_precip": "hourly_scaler_precip.pkl",
        "model": "best_hourly_model.h5",
        "encoder": "hourly_weather_label_encoder.pkl"
    }
    
    found_files = {}
    
    for base_path in possible_paths:
        if not os.path.exists(base_path):
            continue
            
        for file_type, filename in required_files.items():
            if file_type in found_files:
                continue
            full_path = os.path.join(base_path, filename)
            if os.path.exists(full_path):
                found_files[file_type] = full_path
                logger.info(f"Found {file_type}: {full_path}")
    
    return found_files

def load_model_components():
    """Load model components once and cache in memory"""
    global model_components
    
    with model_components['lock']:
        if model_components['loaded']:
            return (
                model_components['model'],
                model_components['scaler'],
                model_components['scaler_precip'],
                model_components['encoder']
            )
        
        try:
            model_files = find_model_files()
            required_types = ["scaler", "scaler_precip", "model", "encoder"]
            missing_files = [f for f in required_types if f not in model_files]
            
            if missing_files:
                raise FileNotFoundError(f"Missing model files: {missing_files}")
            
            # Load scalers
            logger.info("Loading scalers...")
            with open(model_files["scaler"], 'rb') as f:
                scaler = pickle.load(f)
            with open(model_files["scaler_precip"], 'rb') as f:
                scaler_precip = pickle.load(f)
            
            # Load encoder
            logger.info("Loading encoder...")
            with open(model_files["encoder"], 'rb') as f:
                encoder = pickle.load(f)
            
            # Load model
            logger.info("Loading model...")
            try:
                model = load_model(model_files["model"], compile=False)
            except Exception as model_error:
                logger.warning(f"Failed to load model with compile=False: {model_error}")
                import tensorflow.keras.metrics as metrics
                custom_objects = {
                    'mse': metrics.MeanSquaredError,
                    'mae': metrics.MeanAbsoluteError,
                    'accuracy': metrics.CategoricalAccuracy
                }
                model = load_model(model_files["model"], custom_objects=custom_objects, compile=False)
            
            # Cache components
            model_components['model'] = model
            model_components['scaler'] = scaler
            model_components['scaler_precip'] = scaler_precip
            model_components['encoder'] = encoder
            model_components['loaded'] = True
            
            logger.info(f"Encoder classes: {encoder.classes_}")
            logger.info("Model dan preprocessors berhasil dimuat dan di-cache")
            
            return model, scaler, scaler_precip, encoder
            
        except Exception as e:
            logger.error(f"Error loading model/preprocessors: {e}")
            raise

# ================== DATA FETCHING ==================
@lru_cache(maxsize=1)
def fetch_current_weather_cached(cache_key):
    """Cached version of current weather fetching"""
    try:
        response = requests.get(CURRENT_API_URL, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        current = data["current"]
        weather_data = {
            "time": current["time"],
            "temperature_2m": current["temperature_2m"],
            "relative_humidity_2m": current["relative_humidity_2m"], 
            "precipitation": current["precipitation"],
            "cloudcover": current["cloudcover"],
            "windspeed_10m": current["windspeed_10m"],
            "winddirection_10m": current["winddirection_10m"],
            "surface_pressure": current["surface_pressure"],
            "weathercode": current["weathercode"]
        }
        
        logger.info(f"Current weather fetched: {current['time']}")
        return weather_data
        
    except Exception as e:
        logger.error(f"Error fetching current weather: {e}")
        raise

def fetch_current_weather():
    """Fetch current weather with caching per 10 minutes"""
    # Create cache key based on current time (rounded to 10 minutes)
    now = datetime.datetime.utcnow()
    cache_key = now.strftime("%Y%m%d%H") + str(now.minute // 10)
    return fetch_current_weather_cached(cache_key)

def create_historical_data_from_current(current_weather):
    """Create synthetic historical data from current weather"""
    try:
        dummy_data = []
        base_time = datetime.datetime.now() - datetime.timedelta(hours=PAST_HOURS)
        
        # Use known weather codes for dummy data
        known_codes = [0, 1, 2, 3, 51, 53, 61, 63, 65]
        
        for i in range(PAST_HOURS):
            time_stamp = (base_time + datetime.timedelta(hours=i)).strftime('%Y-%m-%dT%H:%M')
            
            # Add small random variations
            temp_var = np.random.normal(0, 2)
            humidity_var = np.random.normal(0, 5)
            
            dummy_record = {
                "time": time_stamp,
                "temperature_2m": current_weather["temperature_2m"] + temp_var,
                "relative_humidity_2m": max(0, min(100, current_weather["relative_humidity_2m"] + humidity_var)),
                "precipitation": max(0, current_weather["precipitation"] + np.random.normal(0, 0.2)),
                "cloudcover": max(0, min(100, current_weather["cloudcover"] + np.random.normal(0, 10))),
                "windspeed_10m": max(0, current_weather["windspeed_10m"] + np.random.normal(0, 1)),
                "winddirection_10m": (current_weather["winddirection_10m"] + np.random.normal(0, 20)) % 360,
                "surface_pressure": current_weather["surface_pressure"] + np.random.normal(0, 3),
                "weathercode": np.random.choice(known_codes)
            }
            dummy_data.append(dummy_record)
        
        return pd.DataFrame(dummy_data)
        
    except Exception as e:
        logger.error(f"Error creating historical data: {e}")
        raise

def clean_dataframe(df):
    """Clean and validate dataframe"""
    try:
        df = df.copy()
        
        # Ensure numeric columns
        numeric_cols = [col for col in NUM_FEATS + ['weathercode'] if col in df.columns]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            if df[col].isnull().any():
                df[col] = df[col].fillna(df[col].mean())
            if df[col].isnull().any():
                df[col] = df[col].fillna(0)
        
        # Remove all-NaN rows
        df = df.dropna(how='all')
        
        # Add missing columns
        required_cols = NUM_FEATS + ['weathercode']
        for col in required_cols:
            if col not in df.columns:
                df[col] = 0
        
        return df
        
    except Exception as e:
        logger.error(f"Error cleaning dataframe: {e}")
        raise

# ================== PREDICTION CORE ==================
def generate_predictions():
    """Generate fresh predictions"""
    try:
        # Load model components
        model, scaler, scaler_precip, encoder = load_model_components()
        
        # Fetch current weather
        current_weather = fetch_current_weather()
        
        # Create historical data (synthetic untuk speed)
        history_df = create_historical_data_from_current(current_weather)
        history_df = clean_dataframe(history_df)
        
        # Add current weather
        current_df = pd.DataFrame([{
            "time": current_weather["time"],
            **{feat: current_weather[feat] for feat in NUM_FEATS},
            "weathercode": current_weather["weathercode"]
        }])
        current_df = clean_dataframe(current_df)
        
        # Combine data
        combined_df = pd.concat([history_df, current_df], ignore_index=True)
        df_recent = combined_df.tail(PAST_HOURS).copy()
        df_recent = clean_dataframe(df_recent)
        
        # Ensure exact length
        if len(df_recent) < PAST_HOURS:
            mean_values = df_recent[NUM_FEATS + ['weathercode']].mean()
            while len(df_recent) < PAST_HOURS:
                new_row = {col: mean_values[col] for col in NUM_FEATS + ['weathercode']}
                new_row['time'] = current_weather["time"]
                df_recent = pd.concat([df_recent, pd.DataFrame([new_row])], ignore_index=True)
        
        # Preprocessing
        df_recent['weathercode'] = df_recent['weathercode'].astype(int)
        df_recent['weather_group'] = df_recent['weathercode'].apply(map_weather_group)
        df_recent['weather_label'] = safe_transform_weather_labels(encoder, df_recent['weather_group'])
        
        # Normalize features
        other_feats = [f for f in NUM_FEATS if f != 'precipitation']
        
        # Ensure all features exist
        for feat in NUM_FEATS:
            if feat not in df_recent.columns:
                df_recent[feat] = 0
            df_recent[feat] = pd.to_numeric(df_recent[feat], errors='coerce').fillna(0)
        
        # Apply scaling
        df_recent[other_feats] = scaler.transform(df_recent[other_feats])
        df_recent[['precipitation']] = scaler_precip.transform(df_recent[['precipitation']])
        
        # Prepare model input
        X_input = df_recent[NUM_FEATS].values.reshape(1, PAST_HOURS, len(NUM_FEATS))
        
        # Validate input
        if np.isnan(X_input).any() or np.isinf(X_input).any():
            raise ValueError("Input contains invalid values")
        
        # Make prediction
        logger.info("Generating predictions...")
        predictions = model.predict(X_input, verbose=0)
        
        # Process predictions
        y_class_pred = predictions[0]
        y_reg_pred = predictions[1]
        
        # Get weather labels
        class_indices = np.argmax(y_class_pred[0], axis=-1)
        weather_labels = encoder.inverse_transform(class_indices)
        
        # Inverse transform regression predictions
        y_reg_predictions = y_reg_pred[0]
        regression_features = [f for f in TARGET_FEATURES if f != 'winddirection_10m']
        
        y_num_combined = np.zeros((FUTURE_HOURS, len(regression_features)))
        
        for i, feat in enumerate(regression_features):
            if feat == 'precipitation':
                y_precip = scaler_precip.inverse_transform(y_reg_predictions[:, [i]])
                y_num_combined[:, i] = np.maximum(y_precip.flatten(), 0)
            else:
                scaler_idx = other_feats.index(feat)
                temp_data = np.zeros((FUTURE_HOURS, len(other_feats)))
                temp_data[:, scaler_idx] = y_reg_predictions[:, i]
                temp_inverse = scaler.inverse_transform(temp_data)
                y_num_combined[:, i] = temp_inverse[:, scaler_idx]
        
        # Build results
        prediction_results = []
        base_time = pd.to_datetime(current_weather["time"])
        
        for i in range(FUTURE_HOURS):
            pred_time = base_time + pd.Timedelta(hours=i+1)
            pred_data = {
                "time": pred_time.strftime("%Y-%m-%dT%H:%M:%S"),
                "weather_label": weather_labels[i],
            }
            
            for j, feat in enumerate(regression_features):
                pred_data[feat] = float(y_num_combined[i, j])
            
            prediction_results.append(pred_data)
        
        result = {
            "current_weather": current_weather,
            "predictions": prediction_results,
            "generated_at": datetime.datetime.now(WITA).strftime("%Y-%m-%dT%H:%M:%S%z")
        }
        
        logger.info(f"Predictions generated successfully for {FUTURE_HOURS} hours")
        return result
        
    except Exception as e:
        logger.error(f"Error generating predictions: {e}")
        raise

def get_cached_predictions():
    """Get predictions from cache or generate new ones"""
    with prediction_cache['lock']:
        now = datetime.datetime.now()
        
        # Check if cache is valid (less than 30 minutes old)
        if (prediction_cache['data'] and prediction_cache['timestamp'] and
            (now - prediction_cache['timestamp']).seconds < 1800):
            logger.info("Returning cached predictions")
            return prediction_cache['data'], True
        
        # Generate new predictions
        logger.info("Generating fresh predictions")
        new_predictions = generate_predictions()
        
        # Update cache
        prediction_cache['data'] = new_predictions
        prediction_cache['timestamp'] = now
        
        return new_predictions, False

# ================== FLASK ROUTES ==================
@app.route('/predict/hourly', methods=['GET'])
def predict_hourly():
    """Main prediction endpoint"""
    try:
        predictions, from_cache = get_cached_predictions()
        
        response = {
            "status": "success",
            "generated_at": predictions["generated_at"],
            "location": {
                "latitude": LATITUDE,
                "longitude": LONGITUDE,
                "city": "Denpasar, Bali"
            },
            "current_weather": predictions["current_weather"],
            "hourly_predictions": predictions["predictions"],
            "metadata": {
                "model_type": "LSTM Multi-output",
                "prediction_horizon_hours": FUTURE_HOURS,
                "features_used": NUM_FEATS,
                "data_source": "cached" if from_cache else "fresh",
                "cache_duration_minutes": 30
            }
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error in prediction endpoint: {e}")
        return jsonify({
            "status": "error",
            "message": str(e),
            "generated_at": datetime.datetime.now(WITA).strftime("%Y-%m-%dT%H:%M:%S%z")
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    try:
        model_files = find_model_files()
        status = "healthy" if len(model_files) == 4 else "unhealthy"
        
        return jsonify({
            "status": status,
            "timestamp": datetime.datetime.now(WITA).strftime("%Y-%m-%dT%H:%M:%S%z"),
            "model_files_found": len(model_files),
            "features_used": NUM_FEATS,
            "location": {
                "latitude": LATITUDE,
                "longitude": LONGITUDE,
                "city": "Denpasar, Bali"
            },
            "cache_status": "active" if prediction_cache['data'] else "empty",
            "model_loaded": model_components['loaded']
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/status', methods=['GET'])
def status():
    """Quick status endpoint"""
    return jsonify({
        "status": "online",
        "service": "Weather Prediction API",
        "timestamp": datetime.datetime.now(WITA).strftime("%Y-%m-%dT%H:%M:%S%z"),
        "version": "2.0-railway"
    })

@app.route('/', methods=['GET'])
def root():
    """Root endpoint"""
    return jsonify({
        "message": "Weather Prediction API",
        "version": "2.0-railway",
        "endpoints": {
            "predictions": "/predict/hourly",
            "health": "/health",
            "status": "/status"
        },
        "timestamp": datetime.datetime.now(WITA).strftime("%Y-%m-%dT%H:%M:%S%z")
    })

# ================== INITIALIZATION ==================
def initialize_app():
    """Initialize application on startup"""
    try:
        logger.info("Initializing Weather Prediction API...")
        
        # Pre-load model components
        load_model_components()
        logger.info("Model components loaded successfully")
        
        # Generate initial predictions
        get_cached_predictions()
        logger.info("Initial predictions generated")
        
        logger.info("Application initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize application: {e}")
        # Don't raise - let the app start anyway for health checks

# ================== MAIN ==================
if __name__ == "__main__":
    # Initialize app
    initialize_app()
    
    # Start Flask app
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)