import json
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd

from backend.config import MODEL_FILES, NEW_WATERSHED_FEATURES, PRIORITY_INPUT_FEATURES
from backend.utils import ensure_model_files


class ModelBundle:
    def __init__(self):
        ensure_model_files()
        self.rainfall_model = joblib.load(MODEL_FILES['rainfall'])
        self.recharge_model = joblib.load(MODEL_FILES['recharge'])
        self.wchi_model = joblib.load(MODEL_FILES['wchi'])
        self.priority_model = joblib.load(MODEL_FILES['priority'])

    def predict_for_features(self, row):
        rainfall_pred = float(self.rainfall_model.predict(pd.DataFrame([row])[list(self.rainfall_model.feature_names_in_)])[0])
        recharge_pred = float(self.recharge_model.predict(pd.DataFrame([row])[list(self.recharge_model.feature_names_in_)])[0])
        wchi_pred = float(self.wchi_model.predict(pd.DataFrame([row])[list(self.wchi_model.feature_names_in_)])[0])
        pr = pd.DataFrame([row])[PRIORITY_INPUT_FEATURES]
        priority_pred = self.priority_model.predict(pr)[0]
        return {
            'Rainfall_Score': rainfall_pred,
            'Recharge_Score': recharge_pred,
            'WCHI_Score': wchi_pred,
            'Priority': priority_pred,
        }


@lru_cache(maxsize=1)
def load_model_bundle():
    return ModelBundle()


def build_prediction_payload(features):
    row = {key: features.get(key, 0) for key in NEW_WATERSHED_FEATURES}
    bundle = load_model_bundle()
    return bundle.predict_for_features(row)


def read_output_csv(name):
    path = Path(__file__).resolve().parent.parent / 'outputs' / name
    if not path.exists():
        return []
    return pd.read_csv(path).to_dict(orient='records')
