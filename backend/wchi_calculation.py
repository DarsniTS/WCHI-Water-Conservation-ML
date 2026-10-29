import numpy as np
import pandas as pd

from backend.config import WEIGHTS


def normalize_component(series):
    s = pd.to_numeric(series, errors='coerce')
    if s.empty:
        return s
    if s.min() == s.max():
        return pd.Series(0.5, index=s.index)
    return (s - s.min()) / (s.max() - s.min())


def compute_wchi_scores(df):
    component_map = {
        'Rainfall': 'Rainfall_Score',
        'Recharge': 'Recharge_Score',
        'Terrain': 'Terrain_Score',
        'Drainage': 'Drainage_Score',
        'Vegetation': 'Vegetation_Score',
        'Groundwater': 'Groundwater_Score',
        'Runoff': 'Runoff_Score',
    }

    subset = df.copy()
    for key, column in component_map.items():
        if column not in subset.columns:
            continue
        subset[f'{key}_norm'] = pd.to_numeric(subset[column], errors='coerce').clip(0, 1)

    for key in component_map:
        subset[f'{key}_norm'] = subset.get(f'{key}_norm', pd.Series(0.5, index=subset.index))

    weighted_terms = []
    for key, weight in WEIGHTS.items():
        col = f'{key}_norm'
        if col in subset.columns:
            weighted_terms.append(subset[col].fillna(0.5) * weight)

    if not weighted_terms:
        return pd.Series(np.zeros(len(df)), index=df.index, name='WCHI_Score')

    wchi = sum(weighted_terms) / sum(WEIGHTS.values())
    wchi = pd.to_numeric(wchi, errors='coerce').clip(0, 1)
    return wchi.rename('WCHI_Score')


def classify_wchi_priority(wchi_values):
    series = pd.Series(wchi_values)
    priority = pd.Series('Low', index=series.index, dtype=object)
    priority.loc[series >= 0.4] = 'Medium'
    priority.loc[series >= 0.7] = 'High'
    return priority
