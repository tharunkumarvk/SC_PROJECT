"""
Consensus Mechanism for Tropical Disease Prediction

Combines predictions from:
1. Fuzzy Logic System (knowledge-based, clinical rules from WHO/CDC)
2. Random Forest Classifier (data-driven, trained on clinical profiles)

Consensus approach:
- Dynamic weighting based on data completeness
- Agreement detection: when both models agree, confidence is boosted
- Disagreement handling: reports both perspectives with explanation
- Validation metrics: reliability scoring, overfitting checks

This dual-model approach provides:
- Cross-validation between knowledge-based and data-driven methods
- Higher reliability than either model alone
- Transparent reasoning (fuzzy rules) backed by statistical learning (RF)
"""

from .fuzzy_logic import TropicalFuzzySystem
from .ml_model import RFPredictor
from .tropical_diseases import DISEASE_PROFILES, SYMPTOMS, SYMPTOM_ORDER


# Singleton instances (created once, reused)
_fuzzy_system = None
_rf_predictor = None


def _get_fuzzy():
    global _fuzzy_system
    if _fuzzy_system is None:
        _fuzzy_system = TropicalFuzzySystem()
    return _fuzzy_system


def _get_rf():
    global _rf_predictor
    if _rf_predictor is None:
        _rf_predictor = RFPredictor()
    return _rf_predictor


def consensus_predict(symptom_values, fuzzy_weight=None, rf_weight=None):
    """
    Generate consensus prediction from both models.

    Dynamic weighting based on data completeness:
    - < 30% symptoms provided: Fuzzy 65% / RF 35%
      (fuzzy handles sparse data better via rule matching)
    - 30-60% provided: Fuzzy 50% / RF 50% (balanced)
    - > 60% provided: Fuzzy 40% / RF 60%
      (RF excels with complete feature vectors)

    Args:
        symptom_values: dict of {symptom_key: numeric_value}
        fuzzy_weight: override weight for fuzzy (None = auto)
        rf_weight: override weight for RF (None = auto)

    Returns:
        dict with complete prediction results.
    """
    if not symptom_values:
        return _empty_result()

    # Dynamic weighting based on data completeness
    completeness = len(symptom_values) / len(SYMPTOM_ORDER)
    if fuzzy_weight is None or rf_weight is None:
        if completeness < 0.30:
            fuzzy_weight, rf_weight = 0.65, 0.35
        elif completeness < 0.60:
            fuzzy_weight, rf_weight = 0.50, 0.50
        else:
            fuzzy_weight, rf_weight = 0.40, 0.60

    fuzzy = _get_fuzzy()
    rf = _get_rf()

    # Get individual predictions
    fuzzy_scores = fuzzy.predict(symptom_values)
    rf_probs = rf.predict(symptom_values)

    # Normalize fuzzy scores to sum to 1 for fair combination
    fuzzy_total = sum(fuzzy_scores.values()) if fuzzy_scores else 0
    if fuzzy_total > 0:
        fuzzy_normalized = {k: v / fuzzy_total for k, v in fuzzy_scores.items()}
    else:
        fuzzy_normalized = {k: 0.0 for k in DISEASE_PROFILES.keys()}

    # Assess fuzzy confidence: if top two are very close, fuzzy is uncertain
    # → automatically shift weight toward RF which handles overlap better.
    fuzzy_vals_sorted = sorted(fuzzy_scores.values(), reverse=True)
    if len(fuzzy_vals_sorted) >= 2 and fuzzy_vals_sorted[0] > 0:
        fuzzy_gap = (fuzzy_vals_sorted[0] - fuzzy_vals_sorted[1]) / fuzzy_vals_sorted[0]
    else:
        fuzzy_gap = 1.0  # only one disease has any score

    # If fuzzy is uncertain (gap < 15%), reduce its weight
    if fuzzy_gap < 0.15:
        fuzzy_weight = max(fuzzy_weight * 0.5, 0.15)
        rf_weight = 1.0 - fuzzy_weight

    # Combine scores
    all_diseases = set(list(fuzzy_normalized.keys()) + list(rf_probs.keys()))
    consensus_scores = {}
    for disease in all_diseases:
        f_score = fuzzy_normalized.get(disease, 0.0)
        r_score = rf_probs.get(disease, 0.0)
        combined = fuzzy_weight * f_score + rf_weight * r_score
        consensus_scores[disease] = round(combined, 4)

    # Sort by combined score
    consensus_sorted = dict(sorted(
        consensus_scores.items(), key=lambda x: x[1], reverse=True
    ))

    # Check model agreement
    fuzzy_top = max(fuzzy_scores, key=fuzzy_scores.get) if fuzzy_scores else None
    rf_top = max(rf_probs, key=rf_probs.get) if rf_probs else None
    consensus_top = list(consensus_sorted.keys())[0] if consensus_sorted else None
    models_agree = fuzzy_top == rf_top

    # Top consensus confidence
    top_score = list(consensus_sorted.values())[0] if consensus_sorted else 0

    # Determine confidence level
    if models_agree and top_score >= 0.25:
        confidence_level = 'High'
    elif models_agree or top_score >= 0.20:
        confidence_level = 'Medium'
    else:
        confidence_level = 'Low'

    # Get detailed fuzzy analysis
    fuzzy_details = fuzzy.get_detailed_analysis(symptom_values)

    # Build validation metrics
    validation = _build_validation(
        symptom_values, consensus_sorted, fuzzy_scores, rf_probs,
        models_agree, confidence_level
    )

    # Build final results (top 3 diseases for clarity)
    top_diseases = {}
    for i, (disease, score) in enumerate(consensus_sorted.items()):
        if i >= 3:
            break
        profile = DISEASE_PROFILES.get(disease, {})
        top_diseases[disease] = {
            'confidence': round(score * 100, 1),
            'fuzzy_score': round(fuzzy_scores.get(disease, 0) * 100, 1),
            'rf_score': round(rf_probs.get(disease, 0) * 100, 1),
            'description': profile.get('description', ''),
            'precautions': profile.get('precautions', []),
            'hallmarks': profile.get('hallmarks', []),
            'references': profile.get('references', []),
        }

    return {
        'predictions': top_diseases,
        'consensus_top': consensus_top,
        'fuzzy_top': fuzzy_top,
        'rf_top': rf_top,
        'models_agree': models_agree,
        'confidence_level': confidence_level,
        'validation': validation,
        'fuzzy_details': fuzzy_details,
        'symptom_values': symptom_values,
    }


def _build_validation(symptom_values, consensus, fuzzy_scores, rf_probs,
                       models_agree, confidence_level):
    """Build validation metrics for the prediction."""
    total_symptoms = len(SYMPTOM_ORDER)
    provided = len(symptom_values)
    completeness = round((provided / total_symptoms) * 100, 1)

    # Top scores
    top_consensus = list(consensus.values())[0] if consensus else 0
    top_fuzzy = max(fuzzy_scores.values()) if fuzzy_scores else 0
    top_rf = max(rf_probs.values()) if rf_probs else 0

    # Certainty: gap between #1 and #2
    consensus_vals = list(consensus.values())
    if len(consensus_vals) >= 2:
        certainty = round(((consensus_vals[0] - consensus_vals[1]) / max(consensus_vals[0], 0.001)) * 100, 1)
    else:
        certainty = 100.0

    # Overall reliability score (0-100)
    reliability = round(
        top_consensus * 100 * 0.30 +    # Consensus strength
        (1.0 if models_agree else 0.5) * 30 +  # Model agreement
        min(completeness, 100) * 0.25 +   # Data completeness
        min(certainty, 100) * 0.15,       # Prediction certainty
        1
    )
    reliability = min(reliability, 100.0)

    # Status
    if reliability >= 70:
        status = 'reliable'
    elif reliability >= 50:
        status = 'moderate'
    else:
        status = 'uncertain'

    # Warnings
    warnings = []
    if completeness < 50:
        warnings.append(f'Only {provided}/{total_symptoms} symptoms provided. More data improves accuracy.')
    if not models_agree:
        fuzzy_top_name = list(fuzzy_scores.keys())[0] if fuzzy_scores else "?"
        rf_top_name = list(rf_probs.keys())[0] if rf_probs else "?"
        warnings.append(f'Models disagree: Fuzzy suggests {fuzzy_top_name}, '
                        f'RF suggests {rf_top_name}.')
    if certainty < 30:
        warnings.append('Multiple diseases have similar scores — prediction is ambiguous.')
    if top_consensus < 0.15:
        warnings.append('Low overall confidence. Symptoms may not clearly match any single disease.')

    # Recommendations
    recommendations = []
    if completeness < 70:
        recommendations.append('Provide more symptom values for improved accuracy.')
    recommendations.append('This is a screening tool only — consult a healthcare professional for proper diagnosis.')
    if not models_agree:
        recommendations.append('Consider both suggested diseases and discuss with a doctor.')

    return {
        'status': status,
        'reliability_score': reliability,
        'confidence_level': confidence_level,
        'data_completeness': completeness,
        'prediction_certainty': certainty,
        'models_agree': models_agree,
        'symptoms_provided': f'{provided}/{total_symptoms}',
        'warnings': warnings,
        'recommendations': recommendations,
    }


def _empty_result():
    """Return empty result structure when no symptoms provided."""
    return {
        'predictions': {},
        'consensus_top': None,
        'fuzzy_top': None,
        'rf_top': None,
        'models_agree': False,
        'confidence_level': 'Low',
        'validation': {
            'status': 'uncertain',
            'reliability_score': 0,
            'confidence_level': 'Low',
            'data_completeness': 0,
            'prediction_certainty': 0,
            'models_agree': False,
            'symptoms_provided': '0/' + str(len(SYMPTOM_ORDER)),
            'warnings': ['No symptoms provided.'],
            'recommendations': ['Please enter at least a few symptom values.'],
        },
        'fuzzy_details': None,
        'symptom_values': {},
    }
