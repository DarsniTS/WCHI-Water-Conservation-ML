import json
from pathlib import Path

import pandas as pd

from backend.config import REQUIRED_COLUMNS, ROOT_DIR


def find_dataset_file():
    from backend.config import DATASET_CANDIDATES

    for path in DATASET_CANDIDATES:
        if path.exists():
            return str(path)
    raise FileNotFoundError(
        'No CSV dataset was found. Expected one of: ' + ', '.join(str(p) for p in DATASET_CANDIDATES)
    )


def validate_dataset(df):
    if df.empty:
        raise ValueError('Dataset is empty.')

    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f'Missing required columns: {missing}')

    if not pd.api.types.is_numeric_dtype(df['Latitude']) or not pd.api.types.is_numeric_dtype(df['Longitude']):
        raise ValueError('Latitude and Longitude must be numeric values.')

    invalid_lat = df[(df['Latitude'] < -90) | (df['Latitude'] > 90)]
    invalid_lon = df[(df['Longitude'] < -180) | (df['Longitude'] > 180)]
    if not invalid_lat.empty:
        raise ValueError('Latitude contains values outside the valid range [-90, 90].')
    if not invalid_lon.empty:
        raise ValueError('Longitude contains values outside the valid range [-180, 180].')

    split_values = set(df['Dataset_Split'].dropna().astype(str).str.strip().unique())
    allowed = {'Train', 'Validation', 'Test'}
    if not split_values.issubset(allowed):
        unexpected = sorted(split_values - allowed)
        raise ValueError(f'Unsupported Dataset_Split values detected: {unexpected}')

    if {'Train', 'Validation', 'Test'} - split_values:
        raise ValueError('Dataset_Split must include Train, Validation, and Test assignments.')

    return True


def detect_column_types(df):
    numeric_cols = []
    categorical_cols = []
    for col in df.columns:
        if col == 'Dataset_Split':
            categorical_cols.append(col)
        elif col in {'Micro_Watershed_ID', 'Date', 'WCHI_Priority', 'Recommended_Conservation_Structure'}:
            categorical_cols.append(col)
        elif pd.api.types.is_numeric_dtype(df[col]):
            numeric_cols.append(col)
        else:
            categorical_cols.append(col)
    return numeric_cols, categorical_cols


def save_json(data, name):
    path = ROOT_DIR / 'outputs' / name
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as fh:
        json.dump(data, fh, indent=2)
    return str(path)


def ensure_model_files():
    from backend.config import MODEL_FILES
    missing = [name for name, path in MODEL_FILES.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(f'Model files missing: {missing}')


def safe_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def make_priority_from_score(score):
    if pd.isna(score):
        return 'Medium'
    if score >= 0.67:
        return 'High'
    if score >= 0.34:
        return 'Medium'
    return 'Low'
