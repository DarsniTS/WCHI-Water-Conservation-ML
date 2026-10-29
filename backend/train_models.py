import json
import os
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.config import DATASET_CANDIDATES, FEATURE_SET, LEAKAGE_EXCLUDED, MODEL_FILES, PRIORITY_INPUT_FEATURES
from backend.recommendation import generate_recommendation_table
from backend.utils import validate_dataset
from backend.wchi_calculation import classify_wchi_priority, compute_wchi_scores

OUTPUTS_DIR = ROOT / 'outputs'
MODELS_DIR = ROOT / 'models'


def resolve_dataset_path():
    for path in DATASET_CANDIDATES:
        if path.exists():
            return path
    raise FileNotFoundError('No dataset file was found in the project data folders.')


def load_and_validate_dataset():
    path = resolve_dataset_path()
    df = pd.read_csv(path)
    print(f'Dataset path: {path}')
    print(f'Dataset shape: {df.shape}')
    validate_dataset(df)
    return df


def split_dataset(df):
    train_df = df[df['Dataset_Split'].astype(str).str.strip() == 'Train'].copy()
    val_df = df[df['Dataset_Split'].astype(str).str.strip() == 'Validation'].copy()
    test_df = df[df['Dataset_Split'].astype(str).str.strip() == 'Test'].copy()
    print(f'Training: {len(train_df)} | Validation: {len(val_df)} | Testing: {len(test_df)}')
    return train_df, val_df, test_df


def create_preprocessor(X):
    numeric_features = X.select_dtypes(include=['number']).columns.tolist()
    categorical_features = [c for c in X.columns if c not in numeric_features]
    transformers = []
    if numeric_features:
        transformers.append(('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), numeric_features))
    if categorical_features:
        transformers.append(('cat', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore'))]), categorical_features))
    return ColumnTransformer(transformers=transformers, remainder='drop')


def evaluate_regression(y_true, y_pred):
    return {
        'MAE': mean_absolute_error(y_true, y_pred),
        'RMSE': mean_squared_error(y_true, y_pred) ** 0.5,
        'R2': r2_score(y_true, y_pred),
    }


def evaluate_classification(y_true, y_pred):
    return {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'Recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'F1': f1_score(y_true, y_pred, average='weighted', zero_division=0),
    }


def train_candidate_regressors(X_train, y_train, X_val, y_val, target_name):
    candidates = {
        'Random Forest Regressor': RandomForestRegressor(n_estimators=300, random_state=42),
        'Gradient Boosting Regressor': GradientBoostingRegressor(random_state=42),
        'Extra Trees Regressor': ExtraTreesRegressor(n_estimators=300, random_state=42),
    }
    results = []
    for label, model in candidates.items():
        pipeline = Pipeline([
            ('preprocess', create_preprocessor(X_train)),
            ('model', model),
        ])
        pipeline.fit(X_train, y_train)
        val_pred = pipeline.predict(X_val)
        metrics = evaluate_regression(y_val, val_pred)
        metrics['Algorithm'] = label
        metrics['Target'] = target_name
        metrics['Split'] = 'Validation'
        results.append((metrics['R2'], metrics))
    results.sort(key=lambda item: item[0], reverse=True)
    return results[0][1], results


def train_candidate_classifiers(X_train, y_train, X_val, y_val, target_name):
    candidates = {
        'Random Forest Classifier': RandomForestClassifier(n_estimators=300, random_state=42),
        'Gradient Boosting Classifier': GradientBoostingClassifier(random_state=42),
    }
    results = []
    for label, model in candidates.items():
        pipeline = Pipeline([
            ('preprocess', create_preprocessor(X_train)),
            ('model', model),
        ])
        pipeline.fit(X_train, y_train)
        val_pred = pipeline.predict(X_val)
        metrics = evaluate_classification(y_val, val_pred)
        metrics['Algorithm'] = label
        metrics['Target'] = target_name
        metrics['Split'] = 'Validation'
        results.append((metrics['F1'], metrics))
    results.sort(key=lambda item: item[0], reverse=True)
    return results[0][1], results


def train_model_for_target(train_df, val_df, test_df, feature_map, target_name):
    feature_cols = feature_map[target_name]
    X_train = train_df[feature_cols].copy()
    X_val = val_df[feature_cols].copy()
    X_test = test_df[feature_cols].copy()
    if target_name == 'WCHI_Priority':
        y_train = classify_wchi_priority(train_df['WCHI_Score'])
        y_val = classify_wchi_priority(val_df['WCHI_Score'])
        y_test = classify_wchi_priority(test_df['WCHI_Score'])
    else:
        y_train = train_df[target_name].copy()
        y_val = val_df[target_name].copy()
        y_test = test_df[target_name].copy()

    if target_name in {'Rainfall_Score', 'Recharge_Score', 'WCHI_Score'}:
        best, _ = train_candidate_regressors(X_train, y_train, X_val, y_val, target_name)
        if best['Algorithm'] == 'Random Forest Regressor':
            model_type = RandomForestRegressor(n_estimators=300, random_state=42)
        elif best['Algorithm'] == 'Gradient Boosting Regressor':
            model_type = GradientBoostingRegressor(random_state=42)
        else:
            model_type = ExtraTreesRegressor(n_estimators=300, random_state=42)
        model = Pipeline([
            ('preprocess', create_preprocessor(X_train)),
            ('model', model_type),
        ])
        model.fit(X_train, y_train)
        test_pred = model.predict(X_test)
        test_metrics = evaluate_regression(y_test, test_pred)
        return model, {'Target': target_name, 'BestAlgorithm': best['Algorithm'], 'Validation': best, 'Test': test_metrics, 'FeatureColumns': feature_cols}

    best, _ = train_candidate_classifiers(X_train, y_train, X_val, y_val, target_name)
    if best['Algorithm'] == 'Random Forest Classifier':
        model_type = RandomForestClassifier(n_estimators=300, random_state=42)
    else:
        model_type = GradientBoostingClassifier(random_state=42)
    model = Pipeline([
        ('preprocess', create_preprocessor(X_train)),
        ('model', model_type),
    ])
    model.fit(X_train, y_train)
    test_pred = model.predict(X_test)
    test_metrics = evaluate_classification(y_test, test_pred)
    return model, {'Target': target_name, 'BestAlgorithm': best['Algorithm'], 'Validation': best, 'Test': test_metrics, 'FeatureColumns': feature_cols}


def save_model(model, name):
    path = MODELS_DIR / f'{name}_model.pkl'
    joblib.dump(model, path)
    print(f'Saved {name} model -> {path}')
    return path


def save_output_csv(df, name):
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUTS_DIR / name
    df.to_csv(path, index=False)
    print(f'Saved output -> {path}')


def build_feature_map():
    return {
        'Rainfall_Score': FEATURE_SET['rainfall'],
        'Recharge_Score': FEATURE_SET['recharge'],
        'WCHI_Score': FEATURE_SET['wchi'],
        'WCHI_Priority': PRIORITY_INPUT_FEATURES,
    }


def generate_prediction_table(df, rainfall_model, recharge_model, wchi_model, priority_model):
    rows = []
    for _, row in df.iterrows():
        rain = row[FEATURE_SET['rainfall']]
        rainfall_pred = float(rainfall_model.predict(pd.DataFrame([rain]))[0])
        recharge = row[FEATURE_SET['recharge']]
        recharge_pred = float(recharge_model.predict(pd.DataFrame([recharge]))[0])
        wchi_val = row[FEATURE_SET['wchi']]
        wchi_pred = float(wchi_model.predict(pd.DataFrame([wchi_val]))[0])
        priority_input = row[PRIORITY_INPUT_FEATURES]
        priority_pred = priority_model.predict(pd.DataFrame([priority_input]))[0]
        rows.append({
            'Micro_Watershed_ID': row['Micro_Watershed_ID'],
            'Latitude': row['Latitude'],
            'Longitude': row['Longitude'],
            'Actual_Rainfall_Score': row['Rainfall_Score'],
            'Predicted_Rainfall_Score': rainfall_pred,
            'Actual_Recharge_Score': row['Recharge_Score'],
            'Predicted_Recharge_Score': recharge_pred,
            'Actual_WCHI_Score': row['WCHI_Score'],
            'Predicted_WCHI_Score': wchi_pred,
            'Actual_WCHI_Priority': classify_wchi_priority(pd.Series([row['WCHI_Score']])).iloc[0],
            'Predicted_WCHI_Priority': priority_pred,
        })
    return pd.DataFrame(rows)


def summarize_model_metrics(results):
    rows = []
    for target_name, item in results.items():
        rows.append({
            'Target': item['Target'],
            'Algorithm': item['BestAlgorithm'],
            'MAE': item['Validation'].get('MAE', ''),
            'RMSE': item['Validation'].get('RMSE', ''),
            'R2': item['Validation'].get('R2', ''),
            'Accuracy': item['Validation'].get('Accuracy', ''),
            'Precision': item['Validation'].get('Precision', ''),
            'Recall': item['Validation'].get('Recall', ''),
            'F1': item['Validation'].get('F1', ''),
            'Validation/Test': 'Validation',
        })
        rows.append({
            'Target': item['Target'],
            'Algorithm': item['BestAlgorithm'],
            'MAE': item['Test'].get('MAE', ''),
            'RMSE': item['Test'].get('RMSE', ''),
            'R2': item['Test'].get('R2', ''),
            'Accuracy': item['Test'].get('Accuracy', ''),
            'Precision': item['Test'].get('Precision', ''),
            'Recall': item['Test'].get('Recall', ''),
            'F1': item['Test'].get('F1', ''),
            'Validation/Test': 'Test',
        })
    return pd.DataFrame(rows)


def main():
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_and_validate_dataset()
    train_df, val_df, test_df = split_dataset(df)
    feature_map = build_feature_map()

    rainfall_model, rainfall_result = train_model_for_target(train_df, val_df, test_df, feature_map, 'Rainfall_Score')
    recharge_model, recharge_result = train_model_for_target(train_df, val_df, test_df, feature_map, 'Recharge_Score')
    wchi_model, wchi_result = train_model_for_target(train_df, val_df, test_df, feature_map, 'WCHI_Score')
    priority_model, priority_result = train_model_for_target(train_df, val_df, test_df, feature_map, 'WCHI_Priority')

    results = {
        'Rainfall_Score': rainfall_result,
        'Recharge_Score': recharge_result,
        'WCHI_Score': wchi_result,
        'WCHI_Priority': priority_result,
    }

    save_model(rainfall_model, 'rainfall')
    save_model(recharge_model, 'recharge')
    save_model(wchi_model, 'wchi')
    save_model(priority_model, 'priority')

    metric_df = summarize_model_metrics(results)
    save_output_csv(metric_df, 'model_metrics.csv')

    df_with_calc = df.copy()
    df_with_calc['WCHI_Score_calc'] = compute_wchi_scores(df_with_calc)
    df_with_calc['WCHI_Priority_calc'] = classify_wchi_priority(df_with_calc['WCHI_Score_calc'])

    prediction_df = generate_prediction_table(df_with_calc, rainfall_model, recharge_model, wchi_model, priority_model)
    save_output_csv(prediction_df, 'predictions.csv')

    recommendation_df = generate_recommendation_table(df_with_calc)
    save_output_csv(recommendation_df, 'recommendations.csv')

    priority_export = df_with_calc[['Micro_Watershed_ID', 'Latitude', 'Longitude', 'WCHI_Score', 'WCHI_Priority']].copy()
    priority_export['Priority_Class'] = df_with_calc['WCHI_Priority_calc']
    save_output_csv(priority_export, 'micro_watershed_priority.csv')

    print('\n=== TRAINING SUMMARY ===')
    print(json.dumps(results, default=str, indent=2))
    print('\n=== MODEL METRICS ===')
    print(metric_df.to_string(index=False))


def refit_leakage_safe_targets():
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    df = load_and_validate_dataset()
    train_df, val_df, test_df = split_dataset(df)
    feature_map = build_feature_map()

    results = {}
    model_names = {'WCHI_Score': 'wchi', 'WCHI_Priority': 'priority'}
    for target_name in ['WCHI_Score', 'WCHI_Priority']:
        model, result = train_model_for_target(train_df, val_df, test_df, feature_map, target_name)
        save_model(model, model_names[target_name])
        results[target_name] = result

    updated_metrics = summarize_model_metrics(results)
    metrics_path = OUTPUTS_DIR / 'model_metrics.csv'
    if metrics_path.exists():
        existing_metrics = pd.read_csv(metrics_path)
        existing_metrics = existing_metrics[~existing_metrics['Target'].isin(updated_metrics['Target'])]
        updated_metrics = pd.concat([existing_metrics, updated_metrics], ignore_index=True)
    save_output_csv(updated_metrics, 'model_metrics.csv')

    prediction_models = {
        name: joblib.load(path)
        for name, path in MODEL_FILES.items()
    }
    prediction_df = generate_prediction_table(
        df,
        prediction_models['rainfall'],
        prediction_models['recharge'],
        prediction_models['wchi'],
        prediction_models['priority'],
    )
    save_output_csv(prediction_df, 'predictions.csv')
    print('\n=== LEAKAGE-SAFE TARGET METRICS ===')
    print(updated_metrics[updated_metrics['Target'].isin(results)].to_string(index=False))


def evaluate_candidate_models():
    df = load_and_validate_dataset()
    train_df, val_df, _ = split_dataset(df)
    feature_map = build_feature_map()
    candidate_rows = []
    for target_name, feature_columns in feature_map.items():
        X_train = train_df[feature_columns].copy()
        X_val = val_df[feature_columns].copy()
        if target_name == 'WCHI_Priority':
            y_train = classify_wchi_priority(train_df['WCHI_Score'])
            y_val = classify_wchi_priority(val_df['WCHI_Score'])
            _, results = train_candidate_classifiers(X_train, y_train, X_val, y_val, target_name)
        else:
            _, results = train_candidate_regressors(
                X_train,
                train_df[target_name].copy(),
                X_val,
                val_df[target_name].copy(),
                target_name,
            )
        for _, metrics in results:
            candidate_rows.append({
                key: value for key, value in metrics.items()
                if key not in {'Split'}
            } | {'Validation/Test': 'Validation Candidate'})

    metric_path = OUTPUTS_DIR / 'model_metrics.csv'
    existing = pd.read_csv(metric_path) if metric_path.exists() else pd.DataFrame()
    if not existing.empty and 'Validation/Test' in existing.columns:
        existing = existing[existing['Validation/Test'] != 'Validation Candidate']
    combined = pd.concat([existing, pd.DataFrame(candidate_rows)], ignore_index=True)
    save_output_csv(combined, 'model_metrics.csv')
    print(pd.DataFrame(candidate_rows).to_string(index=False))


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--refit-leakage-safe',
        action='store_true',
        help='Refit only WCHI and priority models using raw environmental inputs and the existing split.',
    )
    parser.add_argument(
        '--evaluate-candidates',
        action='store_true',
        help='Evaluate configured candidate models on the existing Train/Validation split without saving model artifacts.',
    )
    arguments = parser.parse_args()
    if arguments.refit_leakage_safe:
        refit_leakage_safe_targets()
    elif arguments.evaluate_candidates:
        evaluate_candidate_models()
    else:
        main()
