"""Simple Data Client Implementation for CLM"""
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from enum import Enum


class ModelType(Enum):
    """Model types"""
    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"
    NHITS = "nhits"
    PATCHTST = "patchtst"


class DataClient:
    """Data client protocol"""
    def load_recent_data(self, days: int) -> pd.DataFrame: ...
    def load_validation_data(self, model_type: ModelType) -> pd.DataFrame: ...


class SimpleDataClient(DataClient):
    """Simple file-based data client"""
    
    def __init__(self, data_path: str = "data/binance_training_data_full.csv"):
        self.data_path = Path(data_path)
    
    def load_recent_data(self, days: int) -> pd.DataFrame:
        """Load recent data for drift detection"""
        if not self.data_path.exists():
            return pd.DataFrame()
        
        df = pd.read_csv(self.data_path)
        
        # If timestamp column exists, filter by days
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            cutoff = datetime.now() - timedelta(days=days)
            df = df[df['timestamp'] >= cutoff]
        
        return df
    
    def load_validation_data(self, model_type: ModelType) -> pd.DataFrame:
        """Load validation dataset for model evaluation"""
        # For now, use same data file and take last 20%
        if not self.data_path.exists():
            return pd.DataFrame()
        
        df = pd.read_csv(self.data_path)
        
        # Take last 20% as validation
        split_idx = int(0.8 * len(df))
        return df[split_idx:]
