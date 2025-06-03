import os
import json
import datetime
import calendar
import requests
import pandas as pd
import numpy as np
import pickle
import logging
from tensorflow.keras.models import load_model
from flask import Flask, jsonify
import pytz
import tempfile
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import platform

# Setup logging
log_file = os.path.join(tempfile.gettempdir(), "daily_weather_prediction.log") if platform.system() != "Windows" else os.path.join(os.path.expanduser("~"), "daily_weather_prediction.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Inisialisasi Flask app
app = Flask(__name__)

# Zona waktu WITA (UTC+8)
WITA = pytz.timezone('Asia/Makassar')

# ================== KONFIGURASI ==================
LATITUDE = -8.65
LONGITUDE = 115.22

# Features untuk prediksi
DAILY_FEATS = [
    "temperature_2m_max",
    "temperature_2m_min",
    "precipitation_sum",
    "delta_temp",
    "day_of_month",
    "month"
]

# Regression targets
REG_TARGETS = ['temperature_2m_max', 'temperature_2m_min', 'precipitation_sum']

# API URL untuk Open-Meteo
BASE_API_URL = "https://api.open-meteo.com/v1/forecast"

# File paths menggunakan tempfile
DAILY_HISTORY_CSV = os.path.join(tempfile.gettempdir(), "daily_weather_history.csv")
PAST_DAYS = 30  # Sesuai model input shape
FUTURE_DAYS = 30 # Batasi prediksi ke 7 hari untuk cegah timeout

# Fungsi untuk menemukan file model
def find_model_files():
    """Cari file model di beberapa lokasi dengan logging detail"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    possible_paths = [
        os.path.join(script_dir, ".."),  # Root direktori proyek
        os.path.join(script_dir, "models"),  # Folder models/
        os.path.join(script_dir, "..", "models"),  # Folder models/ di parent
        os.path.join(script_dir, "."),  # Direktori scripts/
        "/app",  # Root di Railway
        "/app/models"  # Folder models/ di Railway
    ]
    
    required_files = {
        "scaler": ["daily_scaler.pkl", "daily_scaler_rule4.pkl"],
        "model": ["best_model_daily.h5", "best_model_rule4.h5"]
    }
    
    found_files = {}
    
    logger.info(f"Current working directory: {os.getcwd()}")
    logger.info(f"Script directory: {script_dir}")
    logger.info("Searching for model files in the following paths:")
    for path in possible_paths:
        logger.info(f"  - {path}")
    
    for base_path in possible_paths:
        abs_path = os.path.abspath(base_path)
        logger.info(f"Checking path: {abs_path}")
        if os.path.exists(abs_path):
            for file_type, filenames in required_files.items():
                if file_type in found_files:
                    continue
                for filename in filenames:
                    full_path = os.path.join(abs_path, filename)
                    logger.info(f"Looking for {file_type} at: {full_path}")
                    if os.path.exists(full_path):
                        found_files[file_type] = full_path
                        logger.info(f"Found {file_type}: {full_path}")
                        break
                    else:
                        logger.warning(f"File not found: {full_path}")
        else:
            logger.warning(f"Path does not exist: {abs_path}")
    
    missing_files = [f for f in required_files if f not in found_files]
    if missing_files:
        logger.error(f"Missing model files: {missing_files}")
        raise FileNotFoundError(f"Missing model files: {missing_files}")
    
    return found_files

# Load model dan preprocessors
def load_model_and_preprocessors():
    """Load model, scaler, dan label encoder"""
    try:
        model_files = find_model_files()
        
        logger.info("Loading scaler and label encoder...")
        with open(model_files["scaler"], 'rb') as f:
            dump = pickle.load(f)
            scaler = dump['scaler']
            label_encoder = dump['label_encoder']
        
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
        
        logger.info(f"Model input shape: {model.input_shape}")
        logger.info(f"Model output shapes: {[output.shape for output in model.outputs]}")
        logger.info(f"Label encoder classes: {label_encoder.classes_}")
        
        return model, scaler, label_encoder
    
    except Exception as e:
        logger.error(f"Error loading model/preprocessors: {e}")
        raise

# Helper untuk API URL
def build_daily_api_url(start_date, end_date, additional_params=None):
    """Buat URL untuk data cuaca harian"""
    base_params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,weather_code",
        "timezone": "Asia/Makassar"
    }
    
    base_params["start_date"] = start_date.strftime('%Y-%m-%d')
    base_params["end_date"] = end_date.strftime('%Y-%m-%d')
    
    if additional_params:
        base_params.update(additional_params)
    
    params_str = "&".join([f"{k}={v}" for k, v in base_params.items()])
    return f"{BASE_API_URL}?{params_str}"

# Mapping weather code
def map_weather_group(code):
    """Map kode cuaca ke grup cuaca"""
    if code in [0, 1, 2, 3]:
        return 'Cerah'
    elif code in [51]:
        return 'Berawan'
    elif code in [53, 55]:
        return 'Gerimis'
    else:
        return 'Hujan'

# Fetch data cuaca
def fetch_daily_weather_data(days_back=PAST_DAYS):
    """Ambil data cuaca harian dari Open-Meteo API"""
    try:
        session = requests.Session()
        retries = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
        session.mount('https://', HTTPAdapter(max_retries=retries))
        
        end_date = datetime.datetime.now(WITA).date()
        start_date = end_date - datetime.timedelta(days=days_back)
        
        api_url = build_daily_api_url(start_date, end_date)
        
        logger.info(f"Fetching daily data from {start_date} to {end_date}")
        response = session.get(api_url, timeout=15)
        response.raise_for_status()
        data = response.json()
        
        if "daily" not in data:
            logger.error("No 'daily' key in API response")
            raise ValueError("Invalid API response format")
        
        daily_data = data["daily"]
        required_fields = ["time", "temperature_2m_max", "temperature_2m_min", "precipitation_sum", "weather_code"]
        missing_fields = [field for field in required_fields if field not in daily_data]
        if missing_fields:
            logger.error(f"Missing fields in API response: {missing_fields}")
            raise ValueError(f"Missing required fields: {missing_fields}")
        
        df = pd.DataFrame({
            "time": daily_data["time"],
            "temperature_2m_max": daily_data["temperature_2m_max"],
            "temperature_2m_min": daily_data["temperature_2m_min"],
            "precipitation_sum": daily_data["precipitation_sum"],
            "weather_code": daily_data["weather_code"],
        })
        
        df['time'] = pd.to_datetime(df['time']).dt.tz_localize(WITA)
        
        for col in ['temperature_2m_max', 'temperature_2m_min', 'weather_code']:
            if df[col].isna().any():
                logger.warning(f"Found NaN values in {col}, filling with mean")
                df[col] = df[col].fillna(df[col].mean())
        
        if df['precipitation_sum'].isna().any():
            logger.warning("Found NaN values in precipitation_sum, filling with 0")
            df['precipitation_sum'] = df['precipitation_sum'].fillna(0.0)
        
        df['delta_temp'] = df['temperature_2m_max'] - df['temperature_2m_min']
        df['day_of_month'] = df['time'].dt.day
        df['month'] = df['time'].dt.month
        df['weather_code'] = pd.to_numeric(df['weather_code'], errors='coerce').fillna(0)
        df['weather_group'] = df['weather_code'].astype(int).apply(map_weather_group)
        
        logger.info(f"Daily weather data berhasil diambil: {len(df)} records")
        return df
        
    except Exception as e:
        logger.error(f"Error fetching daily weather data: {e}")
        return create_dummy_daily_data()

# Dummy data sebagai fallback
def create_dummy_daily_data():
    """Buat data dummy jika API gagal"""
    try:
        dummy_data = []
        base_date = datetime.datetime.now(WITA).date() - datetime.timedelta(days=PAST_DAYS)
        
        for i in range(PAST_DAYS):
            date = base_date + datetime.timedelta(days=i)
            temp_max = 30 + np.random.normal(0, 2)
            temp_min = 24 + np.random.normal(0, 1.5)
            precip = max(0, np.random.exponential(1))
            
            if precip >= 2:
                weather_code = np.random.choice([61, 63, 65])
            elif precip > 0:
                weather_code = np.random.choice([51, 53, 55])
            elif (temp_max - temp_min) < 5:
                weather_code = 51
            else:
                weather_code = np.random.choice([0, 1, 2, 3])
            
            dummy_record = {
                "time": date.strftime('%Y-%m-%d'),
                "temperature_2m_max": temp_max,
                "temperature_2m_min": temp_min,
                "precipitation_sum": precip,
                "weather_code": weather_code
            }
            dummy_data.append(dummy_record)
        
        df = pd.DataFrame(dummy_data)
        df['time'] = pd.to_datetime(df['time']).dt.tz_localize(WITA)
        df['delta_temp'] = df['temperature_2m_max'] - df['temperature_2m_min']
        df['day_of_month'] = df['time'].dt.day
        df['month'] = df['time'].dt.month
        df['weather_group'] = df['weather_code'].astype(int).apply(map_weather_group)
        
        logger.info(f"Dummy daily data created: {len(df)} records")
        return df
        
    except Exception as e:
        logger.error(f"Error creating dummy daily data: {e}")
        raise

# Fungsi prediksi utama
def make_extended_daily_prediction(model, scaler, label_encoder):
    """Lakukan prediksi harian hingga 7 hari"""
    try:
        if not os.path.exists(DAILY_HISTORY_CSV):
            logger.info("CSV file not found, fetching new data...")
            df_daily = fetch_daily_weather_data()
            df_daily.to_csv(DAILY_HISTORY_CSV, index=False)
        else:
            try:
                df_daily = pd.read_csv(DAILY_HISTORY_CSV)
                df_daily['time'] = pd.to_datetime(df_daily['time'])
                if df_daily['time'].dt.tz is None:
                    df_daily['time'] = df_daily['time'].dt.tz_localize(WITA)
                
                last_date = df_daily['time'].max().date()
                today = datetime.datetime.now(WITA).date()
                if (today - last_date).days > 1 or len(df_daily) < PAST_DAYS:
                    logger.info("Updating daily data...")
                    df_daily = fetch_daily_weather_data()
                    df_daily.to_csv(DAILY_HISTORY_CSV, index=False)
            except Exception as csv_error:
                logger.warning(f"Error reading CSV: {csv_error}")
                df_daily = fetch_daily_weather_data()
                df_daily.to_csv(DAILY_HISTORY_CSV, index=False)
        
        # Pastikan cukup data (30 hari)
        if len(df_daily) < PAST_DAYS:
            logger.warning(f"Data kurang dari {PAST_DAYS} hari, mengisi dengan dummy data")
            dummy_df = create_dummy_daily_data()
            df_daily = pd.concat([dummy_df, df_daily]).sort_values('time').reset_index(drop=True)
            df_daily = df_daily.tail(PAST_DAYS)
        
        # Pastikan kolom yang dibutuhkan ada
        required_base_columns = ['temperature_2m_max', 'temperature_2m_min', 'precipitation_sum', 'weather_code', 'time']
        missing_columns = [col for col in required_base_columns if col not in df_daily.columns]
        if missing_columns:
            logger.error(f"Missing required base columns: {missing_columns}")
            raise ValueError(f"Missing required base columns: {missing_columns}")
        
        # Tambah fitur jika belum ada
        if 'delta_temp' not in df_daily.columns:
            df_daily['delta_temp'] = df_daily['temperature_2m_max'] - df_daily['temperature_2m_min']
        if 'day_of_month' not in df_daily.columns:
            df_daily['day_of_month'] = df_daily['time'].dt.day
        if 'month' not in df_daily.columns:
            df_daily['month'] = df_daily['time'].dt.month
        if 'weather_group' not in df_daily.columns:
            df_daily['weather_code'] = pd.to_numeric(df_daily['weather_code'], errors='coerce').fillna(0)
            df_daily['weather_group'] = df_daily['weather_code'].astype(int).apply(map_weather_group)
        
        # Scale fitur dengan nama kolom
        df_daily_scaled = df_daily.copy()
        logger.info(f"Scaling features: {DAILY_FEATS}")
        df_daily_scaled[DAILY_FEATS] = scaler.transform(df_daily_scaled[DAILY_FEATS])
        
        # Tentukan rentang prediksi
        last_date = df_daily['time'].max().date()
        start_date = last_date + datetime.timedelta(days=1)
        end_date = start_date + datetime.timedelta(days=FUTURE_DAYS - 1)
        
        current_weather = {
            "date": last_date.strftime("%Y-%m-%d"),
            "temperature_2m_max": float(df_daily['temperature_2m_max'].iloc[-1]),
            "temperature_2m_min": float(df_daily['temperature_2m_min'].iloc[-1]),
            "precipitation_sum": float(df_daily['precipitation_sum'].iloc[-1]),
            "weather_code": int(df_daily['weather_code'].iloc[-1]),
            "weather_description": df_daily['weather_group'].iloc[-1]
        }
        
        # Siapkan sliding window
        results = []
        df_hist = df_daily_scaled.sort_values('time').reset_index(drop=True)
        last_window = df_hist.tail(PAST_DAYS)[DAILY_FEATS].reset_index(drop=True)
        
        current_date = start_date
        day_count = 0
        
        while current_date <= end_date and day_count < FUTURE_DAYS:
            X_input = last_window.values.reshape(1, PAST_DAYS, len(DAILY_FEATS))
            predictions = model.predict(X_input, verbose=0)
            
            class_probs = predictions[0][0, 0, :]
            class_idx = np.argmax(class_probs)
            desc_pred = label_encoder.inverse_transform([class_idx])[0]
            confidence = float(np.max(class_probs))
            
            reg_scaled = predictions[1][0, 0, :]
            dummy = np.zeros((1, len(DAILY_FEATS)))
            dummy[0, 0] = reg_scaled[0]
            dummy[0, 1] = reg_scaled[1]
            dummy[0, 2] = reg_scaled[2]
            dummy[0, 3] = reg_scaled[0] - reg_scaled[1]
            dummy[0, 4] = current_date.day
            dummy[0, 5] = current_date.month
            
            inv = scaler.inverse_transform(dummy)
            pred_temp_max = float(inv[0, 0])
            pred_temp_min = float(inv[0, 1])
            pred_precip_sum = float(max(0, inv[0, 2]))
            
            results.append({
                'date': current_date.strftime("%Y-%m-%d"),
                'weather_description': desc_pred,
                'confidence': confidence,
                'temperature_2m_max': pred_temp_max,
                'temperature_2m_min': pred_temp_min,
                'precipitation_sum': pred_precip_sum,
                'temperature_range': f"{pred_temp_min:.1f}°C - {pred_temp_max:.1f}°C"
            })
            
            new_feature = pd.DataFrame(
                [[pred_temp_max, pred_temp_min, pred_precip_sum, 
                  pred_temp_max - pred_temp_min, current_date.day, current_date.month]],
                columns=DAILY_FEATS
            )
            new_feature_scaled = scaler.transform(new_feature)[0]
            
            last_window = last_window.drop(index=0).reset_index(drop=True)
            last_window.loc[PAST_DAYS - 1] = new_feature_scaled
            
            current_date += datetime.timedelta(days=1)
            day_count += 1
        
        logger.info(f"Extended prediction completed: {len(results)} days")
        return current_weather, results
        
    except Exception as e:
        logger.error(f"Error in extended daily prediction: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise

# Simpan hasil prediksi ke JSON (tidak dipanggil, untuk fleksibilitas)
def save_daily_predictions_to_json(current_weather, extended_predictions):
    """Simpan hasil prediksi ke JSON di /tmp"""
    try:
        result = {
            "generated_at": datetime.datetime.now(WITA).strftime("%Y-%m-%dT%H:%M:%S%z"),
            "location": {
                "latitude": LATITUDE,
                "longitude": LONGITUDE,
                "city": "Denpasar, Bali"
            },
            "current_weather": current_weather,
            "predictions": extended_predictions,
            "prediction_summary": {
                "total_days_predicted": len(extended_predictions),
                "prediction_start_date": extended_predictions[0]["date"],
                "prediction_end_date": extended_predictions[-1]["date"],
                "covers_full_month": False
            },
            "metadata": {
                "model_type": "LSTM Multi-output Daily Weather Prediction",
                "prediction_horizon": f"Extended Daily ({FUTURE_DAYS} days)",
                "features_used": DAILY_FEATS,
                "weather_categories": ["Berawan", "Cerah", "Gerimis", "Hujan"]
            }
        }
        
        timestamp = datetime.datetime.now(WITA).strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(tempfile.gettempdir(), f"daily_weather_predictions_{timestamp}.json")
        latest_filename = os.path.join(tempfile.gettempdir(), "latest_daily_weather_predictions.json")
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        with open(latest_filename, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Daily predictions saved to {filename}")
        
    except Exception as e:
        logger.error(f"Error saving daily predictions: {e}")
        logger.warning("Continuing without saving JSON files")

# Endpoint utama untuk prediksi
@app.route('/predict', methods=['GET'])
@app.route('/predict/daily', methods=['GET'])
def predict_daily():
    """Endpoint untuk prediksi harian"""
    try:
        model, scaler, label_encoder = load_model_and_preprocessors()
        current_weather, extended_predictions = make_extended_daily_prediction(model, scaler, label_encoder)
        
        response = {
            "status": "success",
            "generated_at": datetime.datetime.now(WITA).strftime("%Y-%m-%dT%H:%M:%S%z"),
            "location": {
                "latitude": LATITUDE,
                "longitude": LONGITUDE,
                "city": "Denpasar, Bali"
            },
            "current_weather": current_weather,
            "predictions": extended_predictions,
            "prediction_summary": {
                "total_days_predicted": len(extended_predictions),
                "prediction_start_date": extended_predictions[0]["date"],
                "prediction_end_date": extended_predictions[-1]["date"],
                "covers_full_month": False
            }
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error in daily prediction endpoint: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return jsonify({
            "status": "error",
            "message": str(e),
            "generated_at": datetime.datetime.now(WITA).strftime("%Y-%m-%dT%H:%M:%S%z")
        }), 500

# Endpoint untuk health check
@app.route('/health', methods=['GET'])
def health_check():
    """Cek kesehatan aplikasi"""
    try:
        model_files = find_model_files()
        status = "healthy" if len(model_files) == 2 else "unhealthy"
        
        return jsonify({
            "status": status,
            "timestamp": datetime.datetime.now(WITA).strftime("%Y-%m-%dT%H:%M:%S%z"),
            "model_files_found": model_files,
            "features_used": DAILY_FEATS,
            "location": {
                "latitude": LATITUDE,
                "longitude": LONGITUDE,
                "city": "Denpasar, Bali"
            }
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

# Endpoint untuk test
@app.route('/test', methods=['GET'])
@app.route('/test/daily', methods=['GET'])
def test_daily():
    """Test endpoint untuk debugging"""
    try:
        output = []
        output.append("=== DAILY PREDICTION DEBUGGING TEST ===")
        output.append(f" Current directory: {os.getcwd()}")
        
        output.append("\n Looking for daily model files...")
        model_files = find_model_files()
        for file_type, path in model_files.items():
            output.append(f"  {file_type}: {path}")
        
        output.append("\n Testing Daily Weather API...")
        try:
            end_date = datetime.datetime.now(WITA).date()
            start_date = end_date - datetime.timedelta(days=7)
            test_url = build_daily_api_url(start_date, end_date)
            
            response = requests.get(test_url, timeout=10)
            output.append(f"  API Status: {response.status_code}")
            output.append(f"  API URL: {test_url}")
            if response.status_code == 200:
                data = response.json()
                output.append(f"  API Data keys: {list(data.keys())}")
                if 'daily' in data:
                    daily_data = data['daily']
                    output.append(f"  Daily data keys: {list(daily_data.keys())}")
                    output.append(f"  Records count: {len(daily_data['time'])}")
                    if 'weather_code' in data['daily']:
                        codes = daily_data['weather_code'][:5]
                        output.append(f"  Sample weather codes: {codes}")
                        mapped = [map_weather_group(int(code)) for code in codes if code is not None]
                        output.append(f"  Mapped weather groups: {mapped}")
            else:
                output.append(f"  API Error: {response.text}")
        except Exception as api_e:
            output.append(f"  API Exception: {api_e}")
        
        output.append(f"\n Features used in training: {DAILY_FEATS}")
        
        output.append("\n Testing data processing...")
        try:
            test_df = fetch_daily_weather_data(days_back=7)
            output.append(f"  Test DataFrame shape: {test_df.shape}")
            output.append(f"  Test DataFrame columns: {test_df.columns.tolist()}")
            output.append(f"  Weather group distribution: {test_df['weather_group'].value_counts().to_dict()}")
        except Exception as data_e:
            output.append(f"  Data processing error: {data_e}")
        
        return jsonify({
            "status": "success",
            "debug_info": "\n".join(output),
            "timestamp": datetime.datetime.now(WITA).strftime("%Y-%m-%dT%H:%M:%S%z")
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)