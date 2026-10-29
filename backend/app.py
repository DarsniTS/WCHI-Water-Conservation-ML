import json
import math
import sys
from functools import lru_cache
from io import BytesIO
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, request, send_file
from flask_cors import CORS

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.config import DATASET_CANDIDATES, FEATURE_SET, MODEL_FILES, NEW_WATERSHED_FEATURES, PRIORITY_INPUT_FEATURES, PRIORITY_ORDER, ROOT_DIR, WEIGHTS
from backend.prediction import load_model_bundle
from backend.recommendation import generate_recommendation_table, normalize, recommend_structures_for_row
from backend.utils import ensure_model_files, find_dataset_file, safe_float
from backend.wchi_calculation import classify_wchi_priority, compute_wchi_scores

app = Flask(__name__)
CORS(app, supports_credentials=True)


@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'API route not found.'}), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'The backend encountered an unexpected error.'}), 500


def dataframe_to_json_records(df):
    if df is None or df.empty:
        return []
    cleaned = df.astype(object).where(pd.notna(df), None)
    return json.loads(cleaned.to_json(orient='records'))


@lru_cache(maxsize=1)
def dataset_df():
    try:
        path = find_dataset_file()
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def watershed_profiles(df):
    if df.empty:
        return pd.DataFrame()
    numeric_columns = df.select_dtypes(include='number').columns.tolist()
    profiles = df.groupby('Micro_Watershed_ID', as_index=False)[numeric_columns].mean()
    if 'Date' in df.columns:
        latest_rows = (
            df.assign(_parsed_date=pd.to_datetime(df['Date'], errors='coerce'))
            .sort_values('_parsed_date')
            .drop_duplicates('Micro_Watershed_ID', keep='last')
        )
    else:
        latest_rows = df.drop_duplicates('Micro_Watershed_ID', keep='last')
    categorical_columns = [
        column for column in df.columns
        if column not in numeric_columns and column != 'Micro_Watershed_ID'
    ]
    if categorical_columns:
        profiles = profiles.merge(
            latest_rows[['Micro_Watershed_ID', *categorical_columns]],
            on='Micro_Watershed_ID',
            how='left',
        )
    if 'WCHI_Score' in profiles.columns:
        profiles['WCHI_Priority'] = classify_wchi_priority(profiles['WCHI_Score']).to_numpy()
        profiles['Priority'] = profiles['WCHI_Priority']
    return profiles


def pipeline_importance(model):
    estimator = model.named_steps.get('model')
    preprocessor = model.named_steps.get('preprocess')
    if estimator is None or preprocessor is None or not hasattr(estimator, 'feature_importances_'):
        return []
    names = preprocessor.get_feature_names_out()
    values = estimator.feature_importances_
    total = float(values.sum())
    importance = [
        {
            'Feature': str(name).split('__', 1)[-1],
            'Importance': round(float(value / total * 100), 2) if total else 0.0,
        }
        for name, value in zip(names, values)
    ]
    return sorted(importance, key=lambda item: item['Importance'], reverse=True)


@app.route('/api/health')
def health():
    dataset_available = False
    try:
        dataset_available = bool(find_dataset_file())
    except FileNotFoundError:
        pass
    models_available = all(path.exists() for path in MODEL_FILES.values())
    predictions_available = (ROOT_DIR / 'outputs' / 'predictions.csv').exists()
    ready = dataset_available and models_available and predictions_available
    return jsonify({
        'status': 'ok' if ready else 'degraded',
        'message': 'WCHI backend is running.',
        'dataset_available': dataset_available,
        'models_available': models_available,
        'predictions_available': predictions_available,
    })


@app.route('/api/dashboard')
def dashboard():
    df = dataset_df()
    if df.empty:
        return jsonify({'error': 'Dataset not found or empty'}), 400
    profiles = watershed_profiles(df)
    prediction_path = ROOT_DIR / 'outputs' / 'predictions.csv'
    predicted = pd.read_csv(prediction_path) if prediction_path.exists() else pd.DataFrame()
    required_prediction_columns = {'Predicted_WCHI_Score', 'Predicted_WCHI_Priority', 'Predicted_Recharge_Score'}
    has_predictions = required_prediction_columns.issubset(predicted.columns)
    if has_predictions:
        wchi = pd.to_numeric(predicted['Predicted_WCHI_Score'], errors='coerce').fillna(0)
        priority = predicted['Predicted_WCHI_Priority'].fillna('Unknown')
        watershed_ids = predicted['Micro_Watershed_ID']
        recharge = pd.to_numeric(predicted['Predicted_Recharge_Score'], errors='coerce').fillna(0)
    else:
        wchi = pd.Series(dtype=float)
        priority = pd.Series(dtype=object)
        watershed_ids = pd.Series(dtype=object)
        recharge = pd.Series(dtype=float)
    recs = generate_recommendation_table(profiles)['Recommended Structure'].fillna('')
    priorities = profiles['WCHI_Priority'] if 'WCHI_Priority' in profiles else pd.Series(dtype=object)
    wchi = pd.to_numeric(profiles.get('WCHI_Score', pd.Series(dtype=float)), errors='coerce')
    rainfall = pd.to_numeric(profiles.get('Rainfall_Score', pd.Series(dtype=float)), errors='coerce')
    recharge = pd.to_numeric(profiles.get('Recharge_Score', pd.Series(dtype=float)), errors='coerce')
    ranking_order = profiles.sort_values('WCHI_Score', ascending=False) if not profiles.empty else profiles
    stats = {
        'total_micro_watersheds': int(len(profiles)),
        'total_observations': int(len(df)),
        'high_priority': int((priorities == 'High').sum()),
        'medium_priority': int((priorities == 'Medium').sum()),
        'low_priority': int((priorities == 'Low').sum()),
        'average_wchi': round(float(wchi.mean()), 4) if not wchi.empty else None,
        'average_rainfall_mm': round(float(pd.to_numeric(profiles.get('Rainfall_mm', pd.Series(dtype=float)), errors='coerce').mean()), 2) if 'Rainfall_mm' in profiles else None,
        'average_rainfall_score': round(float(rainfall.mean()), 4) if not rainfall.empty else None,
        'average_recharge_score': round(float(recharge.mean()), 4) if not recharge.empty else None,
        'highest_wchi_micro_watershed': ranking_order.iloc[0]['Micro_Watershed_ID'] if not ranking_order.empty else None,
        'best_recharge_area': profiles.loc[recharge.idxmax(), 'Micro_Watershed_ID'] if not recharge.empty else None,
        'prediction_available': bool(has_predictions),
        'prediction_rows': int(len(predicted)) if has_predictions else 0,
        'recommended_check_dams': int((recs.astype(str).str.lower().str.contains('check dam|check dam')).sum()) if not recs.empty else 0,
        'recommended_farm_ponds': int((recs.astype(str).str.lower().str.contains('farm pond')).sum()) if not recs.empty else 0,
        'recommended_percolation_ponds': int((recs.astype(str).str.lower().str.contains('percolation pond')).sum()) if not recs.empty else 0,
        'structure_distribution': recs.value_counts().to_dict(),
    }
    return jsonify(stats)


@app.route('/api/analytics/monthly')
def monthly_analytics():
    df = dataset_df()
    if df.empty or 'Date' not in df.columns:
        return jsonify([])
    df = df.copy()
    df['_month'] = pd.to_datetime(df['Date'], errors='coerce').dt.to_period('M').astype(str)
    columns = [column for column in ['Rainfall_mm', 'Recharge_Score', 'Runoff_mm', 'Groundwater_Depth_m', 'WCHI_Score'] if column in df.columns]
    grouped = df.dropna(subset=['_month']).groupby('_month')[columns].mean().reset_index()
    return jsonify(dataframe_to_json_records(grouped.rename(columns={'_month': 'Month'})))


@app.route('/api/micro-watersheds')
def micro_watersheds():
    df = dataset_df()
    if df.empty:
        return jsonify([])
    return jsonify(dataframe_to_json_records(watershed_profiles(df)))


@app.route('/api/micro-watersheds/<id>')
def micro_watershed_detail(id):
    profiles = watershed_profiles(dataset_df())
    if profiles.empty:
        return jsonify({'error': 'Dataset not found'}), 404
    row = profiles[profiles['Micro_Watershed_ID'].astype(str) == str(id)]
    if row.empty:
        return jsonify({'error': 'Micro-watershed not found'}), 404
    return jsonify(dataframe_to_json_records(row)[0])


@app.route('/api/predictions')
def predictions():
    output_path = ROOT_DIR / 'outputs' / 'predictions.csv'
    if not output_path.exists():
        return jsonify([])
    return jsonify(dataframe_to_json_records(pd.read_csv(output_path)))


@app.route('/api/model-metrics')
def model_metrics():
    output_path = ROOT_DIR / 'outputs' / 'model_metrics.csv'
    if not output_path.exists():
        return jsonify([])
    return jsonify(dataframe_to_json_records(pd.read_csv(output_path)))


@app.route('/api/recommendations')
def recommendations():
    df = dataset_df()
    if df.empty:
        return jsonify([])
    return jsonify(dataframe_to_json_records(generate_recommendation_table(watershed_profiles(df))))


@app.route('/api/map')
def map_points():
    df = dataset_df()
    if df.empty:
        return jsonify([])
    subset = watershed_profiles(df)
    records = dataframe_to_json_records(subset)
    for record, (_, row) in zip(records, subset.iterrows()):
        suitability = recommend_structures_for_row(row)
        record['Structure_Suitability'] = suitability
        record['Recommended_Conservation_Structure'] = max(
            suitability, key=lambda item: item['Suitability Score']
        )['Structure']
    return jsonify(records)


@app.route('/api/methodology')
def methodology():
    steps = [
        'Dataset', 'Data Validation', 'Preprocessing', 'Feature Selection', 'Train / Validation / Test',
        'ML Model Training', 'Model Comparison', 'Prediction', 'WCHI Calculation', 'Priority Classification',
        'Suitability Analysis', 'Recommendation', 'GIS Visualization', 'Decision Support'
    ]
    return jsonify({
        'steps': steps,
        'dataset': str(find_dataset_file()),
        'split_counts': dataset_df()['Dataset_Split'].value_counts().to_dict(),
        'raw_model_inputs': NEW_WATERSHED_FEATURES,
        'targets': ['Rainfall_Score', 'Recharge_Score', 'WCHI_Score', 'WCHI_Priority'],
        'wchi_weights': WEIGHTS,
        'priority_thresholds': {'High': 'WCHI >= 0.70', 'Medium': '0.40 <= WCHI < 0.70', 'Low': 'WCHI < 0.40'},
        'explainability_method': 'Tree-model global feature importance; not SHAP or per-site causal attribution.',
    })


@app.route('/api/rankings')
def rankings():
    profiles = watershed_profiles(dataset_df())
    if profiles.empty:
        return jsonify([])
    recommendation_rows = generate_recommendation_table(profiles).set_index('Micro_Watershed_ID')
    profiles['Recommended Structure'] = profiles['Micro_Watershed_ID'].map(recommendation_rows['Recommended Structure'])
    profiles['Suitability Score'] = profiles['Micro_Watershed_ID'].map(recommendation_rows['Suitability Score'])
    priority_filter = request.args.get('priority', 'All')
    search = request.args.get('q', '').strip().lower()
    sort_column = request.args.get('sort', 'WCHI_Score')
    sort_options = {'WCHI_Score', 'Rainfall_Score', 'Recharge_Score', 'Runoff_Score', 'Groundwater_Score', 'Suitability Score'}
    if priority_filter in PRIORITY_ORDER:
        profiles = profiles[profiles['WCHI_Priority'] == priority_filter]
    if search:
        profiles = profiles[profiles['Micro_Watershed_ID'].astype(str).str.lower().str.contains(search)]
    if sort_column not in sort_options:
        sort_column = 'WCHI_Score'
    profiles = profiles.sort_values(sort_column, ascending=False).reset_index(drop=True)
    profiles.insert(0, 'Rank', range(1, len(profiles) + 1))
    return jsonify(dataframe_to_json_records(profiles))


@app.route('/api/model-explainability')
def model_explainability():
    bundle = load_model_bundle()
    targets = {
        'WCHI_Score': bundle.wchi_model,
        'WCHI_Priority': bundle.priority_model,
    }
    return jsonify({
        'method': 'Global tree-based impurity importance. These values describe model-wide use of features, not local causal effects or SHAP values.',
        'targets': {
            target: {
                'model': type(model.named_steps['model']).__name__,
                'features': pipeline_importance(model),
                'input_features': list(model.feature_names_in_),
            }
            for target, model in targets.items()
        },
    })


@app.route('/api/explanations/<id>')
def watershed_explanation(id):
    profiles = watershed_profiles(dataset_df())
    row = profiles[profiles['Micro_Watershed_ID'].astype(str) == str(id)]
    if row.empty:
        return jsonify({'error': 'Micro-watershed not found'}), 404
    profile = row.iloc[0]
    components = [
        {'name': name, 'score': safe_float(profile.get(f'{name}_Score')), 'weight': weight}
        for name, weight in WEIGHTS.items()
    ]
    contributions = [
        {**item, 'weighted_contribution': round(item['score'] * item['weight'] / sum(WEIGHTS.values()), 4)}
        for item in components
    ]
    contributions.sort(key=lambda item: item['weighted_contribution'], reverse=True)
    options = recommend_structures_for_row(profile)
    best = max(options, key=lambda item: item['Suitability Score'])
    return jsonify({
        'Micro_Watershed_ID': id,
        'WCHI_Score': safe_float(profile.get('WCHI_Score')),
        'Priority': profile.get('WCHI_Priority'),
        'priority_explanation': f"Mean observed WCHI is {safe_float(profile.get('WCHI_Score')):.3f}; the configured thresholds classify it as {profile.get('WCHI_Priority')}.",
        'wchi_component_contributions': contributions,
        'recommended_structure': best,
        'structure_method': best['Method'],
    })


@app.route('/api/scenario', methods=['POST'])
def scenario_simulation():
    payload = request.get_json(silent=True) or {}
    watershed_id = str(payload.get('Micro_Watershed_ID', '')).strip()
    components = payload.get('components') or {}
    profiles = watershed_profiles(dataset_df())
    selected = profiles[profiles['Micro_Watershed_ID'].astype(str) == watershed_id]
    if selected.empty:
        return jsonify({'error': 'Select a valid micro-watershed before simulating.'}), 400
    component_names = ['Rainfall', 'Recharge', 'Terrain', 'Drainage', 'Vegetation', 'Groundwater', 'Runoff']
    simulated_components = {}
    for name in component_names:
        key = f'{name}_Score'
        value = components.get(key)
        if value is None:
            return jsonify({'error': f'Missing scenario input: {key}'}), 400
        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            return jsonify({'error': f'{key} must be a number between 0 and 1.'}), 400
        if not 0 <= numeric_value <= 1:
            return jsonify({'error': f'{key} must be between 0 and 1.'}), 400
        simulated_components[key] = numeric_value

    profile = selected.iloc[0].to_dict()
    baseline_score = safe_float(compute_wchi_scores(pd.DataFrame([profile])).iloc[0])
    total_weight = sum(WEIGHTS.values())
    simulated_score = sum(simulated_components[f'{name}_Score'] * weight for name, weight in WEIGHTS.items()) / total_weight
    baseline_recommendations = recommend_structures_for_row(profile)
    simulated_profile = {**profile, **simulated_components, 'WCHI_Score': simulated_score}
    scenario_recommendations = recommend_structures_for_row(simulated_profile)
    baseline_structure = max(baseline_recommendations, key=lambda item: item['Suitability Score'])
    scenario_structure = max(scenario_recommendations, key=lambda item: item['Suitability Score'])
    delta = simulated_score - baseline_score
    return jsonify({
        'Micro_Watershed_ID': watershed_id,
        'notice': 'Scenario simulation only; this is not an observed or forecast environmental measurement.',
        'current': {
            'WCHI_Score': round(baseline_score, 4),
            'Priority': classify_wchi_priority(pd.Series([baseline_score])).iloc[0],
            'Recommended_Structure': baseline_structure['Structure'],
            'Suitability': baseline_structure['Suitability Score'],
        },
        'scenario': {
            'WCHI_Score': round(simulated_score, 4),
            'Priority': classify_wchi_priority(pd.Series([simulated_score])).iloc[0],
            'Recommended_Structure': scenario_structure['Structure'],
            'Suitability': scenario_structure['Suitability Score'],
            'components': simulated_components,
        },
        'difference': round(delta, 4),
        'percentage_change': round(delta / baseline_score * 100, 2) if baseline_score else None,
        'priority_changed': classify_wchi_priority(pd.Series([baseline_score])).iloc[0] != classify_wchi_priority(pd.Series([simulated_score])).iloc[0],
        'suitability_change': round(scenario_structure['Suitability Score'] - baseline_structure['Suitability Score'], 1),
    })


@app.route('/api/future-projection')
def future_projection_status():
    df = dataset_df()
    if 'Date' not in df.columns:
        return jsonify({'available': False, 'reason': 'The dataset does not contain a time field; no future values are generated.'})
    dates = pd.to_datetime(df['Date'], errors='coerce').dropna()
    months = dates.dt.to_period('M').nunique()
    return jsonify({
        'available': False,
        'observed_months': int(months),
        'date_start': dates.min().date().isoformat() if not dates.empty else None,
        'date_end': dates.max().date().isoformat() if not dates.empty else None,
        'reason': f'Only {months} monthly periods across two years are available; this is insufficient to validate a reliable future projection. No future observations are fabricated.',
    })


def validate_prediction_inputs(payload):
    missing = [key for key in NEW_WATERSHED_FEATURES if payload.get(key) is None or payload.get(key) == '']
    if missing:
        return None, {'error': 'Prediction unavailable because required input values are missing.', 'missing_fields': missing}
    feature_row = {}
    invalid = []
    for key in NEW_WATERSHED_FEATURES:
        try:
            value = float(payload[key])
        except (TypeError, ValueError):
            invalid.append(key)
            continue
        if not math.isfinite(value):
            invalid.append(key)
            continue
        feature_row[key] = value
    if invalid:
        return None, {'error': 'Prediction unavailable because some values are not valid finite numbers.', 'invalid_fields': invalid}
    if not feature_row['Month'].is_integer() or not 1 <= feature_row['Month'] <= 12:
        return None, {'error': 'Month must be between 1 and 12.', 'invalid_fields': ['Month']}
    if not feature_row['Year'].is_integer():
        return None, {'error': 'Year must be a whole number.', 'invalid_fields': ['Year']}
    if not -90 <= feature_row['Latitude'] <= 90:
        return None, {'error': 'Latitude must be between -90 and 90.', 'invalid_fields': ['Latitude']}
    if not -180 <= feature_row['Longitude'] <= 180:
        return None, {'error': 'Longitude must be between -180 and 180.', 'invalid_fields': ['Longitude']}
    for key in ['Sand_pct', 'Clay_pct', 'Silt_pct', 'Agriculture_pct', 'Forest_pct', 'Builtup_pct', 'Waterbody_pct', 'Barren_Land_pct', 'Relative_Humidity_pct']:
        if not 0 <= feature_row[key] <= 100:
            return None, {'error': f'{key} must be between 0 and 100.', 'invalid_fields': [key]}
    return feature_row, None


def predict_feature_row(feature_row, observed=None):
    bundle = load_model_bundle()
    observed = observed if observed is not None else dataset_df()
    rainfall = float(bundle.rainfall_model.predict(pd.DataFrame([feature_row])[FEATURE_SET['rainfall']])[0])
    recharge = float(bundle.recharge_model.predict(pd.DataFrame([feature_row])[FEATURE_SET['recharge']])[0])
    wchi = float(bundle.wchi_model.predict(pd.DataFrame([feature_row])[FEATURE_SET['wchi']])[0])
    priority_input = pd.DataFrame([feature_row])[PRIORITY_INPUT_FEATURES]
    model_priority = str(bundle.priority_model.predict(priority_input)[0])
    threshold_priority = str(classify_wchi_priority(pd.Series([wchi])).iloc[0])
    confidence = None
    if hasattr(bundle.priority_model, 'predict_proba'):
        probabilities = bundle.priority_model.predict_proba(priority_input)[0]
        class_index = list(bundle.priority_model.classes_).index(threshold_priority)
        confidence = round(float(probabilities[class_index]) * 100, 1)

    def observed_scale(column, value):
        series = pd.to_numeric(observed[column], errors='coerce').dropna()
        if series.empty:
            return 0.5
        return normalize(value, float(series.min()), float(series.max()))

    recommendation_row = {
        **feature_row,
        'Rainfall_Score': rainfall,
        'Recharge_Score': recharge,
        'Terrain_Score': 1 - observed_scale('Mean_Slope_deg', feature_row['Mean_Slope_deg']),
        'Drainage_Score': observed_scale('Drainage_Density_km_per_km2', feature_row['Drainage_Density_km_per_km2']),
        'Vegetation_Score': (observed_scale('NDVI', feature_row['NDVI']) + observed_scale('Forest_pct', feature_row['Forest_pct'])) / 2,
        'Groundwater_Score': 1 - observed_scale('Groundwater_Depth_m', feature_row['Groundwater_Depth_m']),
        'Runoff_Score': observed_scale('Runoff_mm', feature_row['Runoff_mm']),
        'WCHI_Score': wchi,
    }
    recommendation_options = recommend_structures_for_row(recommendation_row)
    recommendation = max(recommendation_options, key=lambda item: item['Suitability Score'])
    return {
        'Predicted Rainfall Score': round(rainfall, 4),
        'Predicted Recharge Score': round(recharge, 4),
        'Predicted WCHI Score': round(wchi, 4),
        'Priority': threshold_priority,
        'Priority Model Prediction': model_priority,
        'Priority Model Confidence': confidence,
        'Priority Model Confidence Note': 'Uncalibrated classifier score; use as a model diagnostic, not a measured probability.',
        'Recommended Structure': recommendation['Structure'],
        'Structure Suitability': recommendation['Suitability Score'],
        'Suitability Level': recommendation['Suitability Level'],
        'Recommendation Reason': recommendation['Reason'],
        'Supporting Factors': recommendation['Supporting Factors'],
        'Limiting Factors': recommendation['Limiting Factors'],
        'Recommendation Method': recommendation['Method'],
        'Structure Suitability Options': recommendation_options,
        'Data Quality': 'Complete',
    }


@app.route('/api/prediction-defaults')
def prediction_defaults():
    df = dataset_df()
    if df.empty:
        return jsonify({'error': 'Dataset not found or empty'}), 400
    defaults = {}
    for key in NEW_WATERSHED_FEATURES:
        values = pd.to_numeric(df[key], errors='coerce').dropna()
        if values.empty:
            return jsonify({'error': f'No valid dataset defaults are available for {key}.'}), 500
        defaults[key] = float(values.median())
    defaults['Month'] = int(round(defaults['Month']))
    defaults['Year'] = int(round(defaults['Year']))
    return jsonify({'values': defaults, 'source': 'Median of the existing project dataset; editable before prediction.'})


@app.route('/api/predict', methods=['POST'])
def predict_new_watershed():
    payload = request.get_json(silent=True) or {}
    feature_row, validation_error = validate_prediction_inputs(payload)
    if validation_error:
        return jsonify(validation_error), 400
    return jsonify(predict_feature_row(feature_row))


@app.route('/api/batch-predict', methods=['POST'])
def batch_predict():
    uploaded = request.files.get('file')
    if uploaded is None or not uploaded.filename:
        return jsonify({'error': 'Choose a CSV file to upload.'}), 400
    if Path(uploaded.filename).suffix.lower() != '.csv':
        return jsonify({'error': 'Only CSV files are supported.'}), 400
    try:
        input_df = pd.read_csv(uploaded)
    except Exception:
        return jsonify({'error': 'The uploaded file could not be read as a CSV.'}), 400
    missing_columns = [key for key in NEW_WATERSHED_FEATURES if key not in input_df.columns]
    if missing_columns:
        return jsonify({'error': 'CSV is missing required model input columns.', 'missing_columns': missing_columns}), 400
    if input_df.empty:
        return jsonify({'error': 'The uploaded CSV has no prediction rows.'}), 400
    outputs = []
    observed = dataset_df()
    for row_number, (_, row) in enumerate(input_df.iterrows(), start=2):
        feature_row, validation_error = validate_prediction_inputs(row.to_dict())
        if validation_error:
            return jsonify({**validation_error, 'row': row_number}), 400
        outputs.append({**row.to_dict(), **predict_feature_row(feature_row, observed)})
    csv_data = pd.DataFrame(outputs).to_csv(index=False).encode('utf-8')
    return send_file(
        BytesIO(csv_data),
        as_attachment=True,
        download_name='wchi_batch_predictions.csv',
        mimetype='text/csv',
    )


@app.route('/api/download/predictions')
def download_predictions():
    path = ROOT_DIR / 'outputs' / 'predictions.csv'
    if not path.exists():
        return jsonify({'error': 'No predictions output found'}), 404
    return send_file(path, as_attachment=True, download_name='predictions.csv')


@app.route('/api/download/recommendations')
def download_recommendations():
    df = dataset_df()
    if df.empty:
        return jsonify({'error': 'Dataset not found or empty'}), 404
    csv_data = generate_recommendation_table(df).to_csv(index=False).encode('utf-8')
    return send_file(
        BytesIO(csv_data),
        as_attachment=True,
        download_name='recommendations.csv',
        mimetype='text/csv',
    )


@app.route('/api/download/metrics')
def download_metrics():
    path = ROOT_DIR / 'outputs' / 'model_metrics.csv'
    if not path.exists():
        return jsonify({'error': 'No metrics output found'}), 404
    return send_file(path, as_attachment=True, download_name='model_metrics.csv')


@app.route('/api/download/priorities')
def download_priorities():
    path = ROOT_DIR / 'outputs' / 'micro_watershed_priority.csv'
    if not path.exists():
        return jsonify({'error': 'No priority output found'}), 404
    return send_file(path, as_attachment=True, download_name='micro_watershed_priority.csv')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
