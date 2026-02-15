import pandas as pd
import numpy as np
import streamlit as st
import psutil
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from config import DATA_DIR, DATA_FILES, SELECTED_COLUMNS, TARGET_COLUMN, RAM_WARNING_THRESHOLD
from utils.logger import setup_logger

logger = setup_logger()

def get_system_metrics():
    """
    Returns current RAM and CPU usage.
    """
    ram = psutil.virtual_memory()
    cpu = psutil.cpu_percent(interval=None)
    return {
        "ram_percent": ram.percent,
        "ram_used_gb": round(ram.used / (1024**3), 2),
        "ram_total_gb": round(ram.total / (1024**3), 2),
        "cpu_percent": cpu
    }

@st.cache_data(show_spinner=False)
def load_data(sample_size=None):
    """
    Load data from CSVs with optional sampling to save memory.
    Cached by Streamlit.
    """
    dfs = []
    
    # Check RAM before loading
    metrics = get_system_metrics()
    if metrics["ram_percent"] > RAM_WARNING_THRESHOLD:
        st.warning(f"⚠️ High Memory Usage detected ({metrics['ram_percent']}%)! Enforcing strict sampling.")
        sample_size = min(sample_size or 50000, 50000)

    try:
        for file in DATA_FILES:
            file_path = DATA_DIR / file
            if file_path.exists():
                logger.info(f"Loading {file}...")
                
                # If sampling is requested, we read with nrows first to save IO/memory
                if sample_size:
                    # Heuristic: Read more than needed then sample to ensure randomness
                    rows_to_read = max(200000, sample_size * 2) 
                    df = pd.read_csv(file_path, usecols=SELECTED_COLUMNS, nrows=rows_to_read)
                    df = df.sample(n=min(len(df), sample_size // len(DATA_FILES)), random_state=42)
                else:
                    df = pd.read_csv(file_path, usecols=SELECTED_COLUMNS)
                
                # Optimization
                for col in df.select_dtypes(include=['float64']).columns:
                    df[col] = df[col].astype('float32')
                for col in df.select_dtypes(include=['int64']).columns:
                    df[col] = df[col].astype('int32')

                dfs.append(df)
            else:
                logger.warning(f"File not found: {file}")
        
        if not dfs:
            raise FileNotFoundError("No dataset files found.")
            
        full_df = pd.concat(dfs, ignore_index=True)
        # Drop duplicates across all files
        full_df.drop_duplicates(inplace=True)
        logger.info(f"Data loaded successfully. Shape: {full_df.shape}")
        
        return full_df
        
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        st.error(f"Failed to load data: {e}")
        return pd.DataFrame()

@st.cache_data(show_spinner=False)
def preprocess_data(df, target_col=TARGET_COLUMN):
    """
    Clean, Encode, and Scale the data.
    Returns: X, y, label_encoder_target, scaler
    """
    try:
        logger.info("Starting preprocessing...")
        df = df.copy()
        
        # Separate X and y
        X = df.drop(columns=[target_col])
        y = df[target_col]

        # Handle Missing Values (Imputation)
        num_cols = X.select_dtypes(include=['number']).columns
        if len(num_cols) > 0:
            imputer = SimpleImputer(strategy='mean')
            X[num_cols] = imputer.fit_transform(X[num_cols])

        # Categorical Encoding (Features)
        # We need to handle potential string columns
        cat_cols = X.select_dtypes(exclude=['number']).columns
        for col in cat_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))

        # Target Encoding
        le_target = LabelEncoder()
        y = le_target.fit_transform(y.astype(str))

        # Scaling
        scaler = StandardScaler()
        X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

        return X_scaled, y, le_target, scaler

    except Exception as e:
        logger.error(f"Preprocessing failed: {e}")
        st.error(f"Preprocessing Error: {e}")
        return None, None, None, None

def split_data(X, y, test_size=0.3, random_state=42):
    """
    Splits data into training and testing sets.
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
