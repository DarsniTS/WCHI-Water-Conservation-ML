import numpy as np
import pandas as pd


def normalize(value, low, high):
    if pd.isna(value):
        return 0.0
    if high == low:
        return 0.5
    return max(0.0, min(1.0, (value - low) / (high - low)))


def recommend_structures_for_row(row):
    wchi = float(row.get('WCHI_Score', 0.0) or 0.0)
    rainfall = float(row.get('Rainfall_Score', 0.0) or 0.0)
    recharge = float(row.get('Recharge_Score', 0.0) or 0.0)
    terrain = float(row.get('Terrain_Score', 0.0) or 0.0)
    drainage = float(row.get('Drainage_Score', 0.0) or 0.0)
    vegetation = float(row.get('Vegetation_Score', 0.0) or 0.0)
    groundwater = float(row.get('Groundwater_Score', 0.0) or 0.0)
    runoff = float(row.get('Runoff_Score', 0.0) or 0.0)
    slope_degrees = float(row.get('Mean_Slope_deg', 0.0) or 0.0)
    slope = normalize(slope_degrees, 0, 45)
    drainage_density = float(row.get('Drainage_Density_km_per_km2', 0.0) or 0.0)
    infiltration = normalize(float(row.get('Infiltration_Rate_mm_hr', 0.0) or 0.0), 0, 20)
    permeability = float(row.get('Geological_Permeability_Index', 0.0) or 0.0)
    existing_ponds = float(row.get('Existing_Ponds', 0.0) or 0.0)
    existing_check_dams = float(row.get('Existing_Check_Dams', 0.0) or 0.0)
    agriculture = normalize(float(row.get('Agriculture_pct', 0.0) or 0.0), 0, 100)
    forest = normalize(float(row.get('Forest_pct', 0.0) or 0.0), 0, 100)
    barren = normalize(float(row.get('Barren_Land_pct', 0.0) or 0.0), 0, 100)
    runoff_mm = float(row.get('Runoff_mm', 0.0) or 0.0)
    groundwater_depth = float(row.get('Groundwater_Depth_m', 0.0) or 0.0)

    recommendations = []

    check_dam_score = 0.35 * runoff + 0.25 * drainage + 0.20 * terrain + 0.20 * wchi
    if slope_degrees > 5 and drainage_density > 1.5 and runoff_mm > 10:
        check_dam_score += 0.2
    if existing_check_dams > 0:
        check_dam_score -= 0.15
    check_dam_score = max(0, min(1, check_dam_score))
    recommendations.append({
        'Structure': 'Check Dam',
        'Suitability Score': round(check_dam_score * 100, 1),
        'Suitability Level': 'Highly Suitable' if check_dam_score >= 0.7 else 'Moderately Suitable' if check_dam_score >= 0.45 else 'Less Suitable',
        'Reason': 'High runoff, good drainage connectivity, and suitable terrain support storage and recharge in the watershed.'
    })

    farm_pond_score = 0.30 * rainfall + 0.25 * runoff + 0.25 * wchi + 0.20 * agriculture
    if runoff_mm > 8 and groundwater_depth > 2:
        farm_pond_score += 0.15
    farm_pond_score = max(0, min(1, farm_pond_score))
    recommendations.append({
        'Structure': 'Farm Pond',
        'Suitability Score': round(farm_pond_score * 100, 1),
        'Suitability Level': 'Highly Suitable' if farm_pond_score >= 0.7 else 'Moderately Suitable' if farm_pond_score >= 0.45 else 'Less Suitable',
        'Reason': 'Moderate-to-high runoff and agricultural land use indicate good potential for local water harvesting and storage.'
    })

    percolation_score = 0.30 * recharge + 0.25 * runoff + 0.20 * infiltration + 0.25 * permeability
    if groundwater_depth > 4 and existing_ponds == 0:
        percolation_score += 0.1
    percolation_score = max(0, min(1, percolation_score))
    recommendations.append({
        'Structure': 'Percolation Pond',
        'Suitability Score': round(percolation_score * 100, 1),
        'Suitability Level': 'Highly Suitable' if percolation_score >= 0.7 else 'Moderately Suitable' if percolation_score >= 0.45 else 'Less Suitable',
        'Reason': 'High recharge potential, strong infiltration conditions, and permeable geology make this structure effective for groundwater augmentation.'
    })

    recharge_structure_score = 0.30 * recharge + 0.25 * groundwater + 0.20 * infiltration + 0.25 * permeability
    if groundwater_depth > 3 and barren < 25:
        recharge_structure_score += 0.1
    recharge_structure_score = max(0, min(1, recharge_structure_score))
    recommendations.append({
        'Structure': 'Recharge Structure / Recharge Pit',
        'Suitability Score': round(recharge_structure_score * 100, 1),
        'Suitability Level': 'Highly Suitable' if recharge_structure_score >= 0.7 else 'Moderately Suitable' if recharge_structure_score >= 0.45 else 'Less Suitable',
        'Reason': 'Sufficient recharge potential and permeable strata suggest the site can benefit from recharge pits and infiltration enhancement.'
    })

    contour_score = 0.35 * terrain + 0.25 * slope + 0.20 * vegetation + 0.20 * runoff
    if slope_degrees > 9 and float(row.get('Barren_Land_pct', 0.0) or 0.0) > 20:
        contour_score += 0.15
    contour_score = max(0, min(1, contour_score))
    recommendations.append({
        'Structure': 'Contour Bund',
        'Suitability Score': round(contour_score * 100, 1),
        'Suitability Level': 'Highly Suitable' if contour_score >= 0.7 else 'Moderately Suitable' if contour_score >= 0.45 else 'Less Suitable',
        'Reason': 'Steeper slopes and erosion-prone land conditions support contour bunding to reduce runoff and improve soil moisture retention.'
    })

    vegetation_score = 0.35 * vegetation + 0.25 * wchi + 0.20 * forest + 0.20 * barren
    vegetation_score = max(0, min(1, vegetation_score))
    recommendations.append({
        'Structure': 'Vegetative / Soil Conservation',
        'Suitability Score': round(vegetation_score * 100, 1),
        'Suitability Level': 'Highly Suitable' if vegetation_score >= 0.7 else 'Moderately Suitable' if vegetation_score >= 0.45 else 'Less Suitable',
        'Reason': 'Low vegetation cover and exposed land increase the relevance of vegetative filtering and soil conservation measures.'
    })

    factor_context = {
        'Check Dam': {
            'support': [
                (runoff >= 0.5 or runoff_mm > 10, 'Elevated runoff supports temporary storage.'),
                (drainage_density > 1.5 and drainage >= 0.5, 'Drainage connectivity supports water capture.'),
                (5 <= slope_degrees <= 15, 'Moderate slope is within the rule set preferred range.'),
            ],
            'limits': [
                (existing_check_dams > 0, 'Existing check dams may reduce the need for additional structures.'),
                (drainage_density <= 1.5, 'Low drainage density may limit catchment inflow.'),
                (slope_degrees > 20, 'Steep terrain needs site-level stability assessment.'),
            ],
        },
        'Farm Pond': {
            'support': [
                (agriculture >= 0.4, 'Agricultural land use can benefit from local water storage.'),
                (rainfall >= 0.5, 'Rainfall conditions support seasonal collection.'),
                (runoff_mm > 8, 'Available runoff can contribute to pond inflow.'),
            ],
            'limits': [
                (existing_ponds > 0, 'Existing ponds may reduce the need for additional storage.'),
                (float(row.get('Builtup_pct', 0.0) or 0.0) > 30, 'High built-up coverage may constrain suitable land.'),
            ],
        },
        'Percolation Pond': {
            'support': [
                (recharge >= 0.5, 'Recharge conditions support infiltration storage.'),
                (infiltration >= 0.5, 'Infiltration rate supports percolation.'),
                (permeability >= 0.5, 'Geological permeability supports groundwater movement.'),
            ],
            'limits': [
                (existing_ponds > 0, 'Existing ponds may already provide local percolation capacity.'),
                (infiltration < 0.25, 'Low infiltration may limit percolation performance.'),
                (permeability < 0.25, 'Low permeability may limit recharge.'),
            ],
        },
        'Recharge Structure / Recharge Pit': {
            'support': [
                (recharge >= 0.5, 'Recharge score supports groundwater augmentation.'),
                (groundwater >= 0.5, 'Groundwater conditions contribute positively to the suitability score.'),
                (infiltration >= 0.5 and permeability >= 0.5, 'Infiltration and permeability support recharge.'),
            ],
            'limits': [
                (infiltration < 0.25, 'Low infiltration may limit recharge performance.'),
                (permeability < 0.25, 'Low permeability may restrict subsurface movement.'),
                (groundwater_depth <= 2, 'Shallow groundwater requires site-specific review.'),
            ],
        },
        'Contour Bund': {
            'support': [
                (slope_degrees >= 5, 'Slope provides a runoff-control opportunity.'),
                (float(row.get('Barren_Land_pct', 0.0) or 0.0) > 20, 'Exposed land increases the relevance of erosion control.'),
                (vegetation < 0.5, 'Lower vegetation score supports soil-moisture conservation measures.'),
            ],
            'limits': [
                (slope_degrees < 2, 'Very gentle slope may limit contour-bund benefit.'),
                (slope_degrees > 20, 'Steep terrain needs engineering and stability review.'),
            ],
        },
        'Vegetative / Soil Conservation': {
            'support': [
                (float(row.get('Barren_Land_pct', 0.0) or 0.0) > 20, 'Exposed land supports soil-conservation measures.'),
                (float(row.get('NDVI', 0.0) or 0.0) < 0.35, 'Lower vegetation cover supports revegetation measures.'),
                (slope_degrees >= 5, 'Slope increases the value of erosion-control measures.'),
            ],
            'limits': [
                (float(row.get('Forest_pct', 0.0) or 0.0) > 35, 'Existing forest cover may reduce the need for revegetation.'),
            ],
        },
    }
    for recommendation in recommendations:
        factors = factor_context[recommendation['Structure']]
        recommendation['Supporting Factors'] = [text for condition, text in factors['support'] if condition]
        recommendation['Limiting Factors'] = [text for condition, text in factors['limits'] if condition]
        if not recommendation['Supporting Factors']:
            recommendation['Supporting Factors'] = ['No strong supporting threshold was met by the current rule set.']
        if not recommendation['Limiting Factors']:
            recommendation['Limiting Factors'] = ['No limiting threshold was triggered by the current rule set.']
        recommendation['Method'] = 'Rule-based suitability; not an ML structure classifier.'

    return recommendations


def generate_recommendation_table(df):
    results = []
    for _, row in df.iterrows():
        recs = recommend_structures_for_row(row)
        top = max(recs, key=lambda item: item['Suitability Score'])
        results.append({
            'Micro_Watershed_ID': row.get('Micro_Watershed_ID', ''),
            'Priority': row.get('WCHI_Priority', 'Unknown'),
            'WCHI': round(float(row.get('WCHI_Score', 0.0) or 0.0), 4),
            'Recommended Structure': top['Structure'],
            'Suitability': f"{top['Suitability Score']:.1f}%",
            'Suitability Score': top['Suitability Score'],
            'Suitability Level': top['Suitability Level'],
            'Reason': top['Reason'],
            'Supporting Factors': '; '.join(top['Supporting Factors']),
            'Limiting Factors': '; '.join(top['Limiting Factors']),
            'Method': top['Method'],
        })
    return pd.DataFrame(results)
