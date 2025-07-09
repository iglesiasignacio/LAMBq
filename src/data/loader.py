import pandas as pd
import logging

from src.constants import CUE_COL

def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    data = data.loc[~data[CUE_COL].isna()].reset_index(drop=True)
    return data

def load_data(file_path: str) -> pd.DataFrame:
    try:
        data = pd.read_csv(file_path)
        data = clean_data(data)
        logging.info(f"Data loaded successfully from {file_path}")
        return data
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        return pd.DataFrame()
