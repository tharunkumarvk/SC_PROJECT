"""
Fuzzy Logic Engine for Tropical Disease Prediction

Uses membership functions calibrated to clinical symptom ranges.
Rules are derived from WHO/CDC clinical profiles (see tropical_diseases.py).

Membership function design:
- Fever (°F): 5-level fuzzy sets mapped to standard medical thresholds
  Normal: <98.6°F, Low-grade: 98.6-100.4°F, Moderate: 100.4-102°F,
  High: 102-104°F, Very High: >104°F
  (Based on standard clinical temperature classification)
- Severity (0-10): 5-level fuzzy sets with overlapping triangular/
  trapezoidal functions for smooth transitions

The fuzzy system evaluates how well a patient's symptoms match each
disease's expected pattern, using weighted membership matching.
"""

from .tropical_diseases import (
    SYMPTOMS, SYMPTOM_ORDER, DISEASE_PROFILES,
    FUZZY_DISEASE_RULES, get_disease_names
)


# ─── Membership Functions ─────────────────────────────────────────────────────

def _triangular(x, a, b, c):
    """Triangular membership function: peak=1.0 at b, zero at a and c.
    Uses strict < at boundaries so peak value always gets membership > 0."""
    if x < a or x > c:
        return 0.0
    if abs(x - b) < 1e-9:
        return 1.0  # exact peak
    if a == b:
        return (c - x) / (c - b) if c != b else 0.0
    if b == c:
        return (x - a) / (b - a) if b != a else 0.0
    if x <= b:
        return (x - a) / (b - a)
    return (c - x) / (c - b)


def _trapezoidal(x, a, b, c, d):
    """Trapezoidal membership function: plateau=1.0 between b and c.
    At x=a returns 0, at x=b returns 1, at x=c returns 1, at x=d returns 0.
    Inclusive boundaries: x=a and x=d return small epsilon for continuity."""
    if x < a or x > d:
        return 0.0
    if b <= x <= c:
        return 1.0
    if x <= a:  # x == a edge
        return 0.01  # small nonzero for boundary continuity
    if x >= d:  # x == d edge
        return 0.01
    if x < b:
        return (x - a) / (b - a) if b != a else 1.0
    return (d - x) / (d - c) if d != c else 1.0


# ─── Fever Fuzzy Sets (°F) ───────────────────────────────────────────────────
# Standard medical temperature classification with OVERLAPPING regions:
#   Normal: < 99°F  |  Low-grade: 98-100.4°F  |  Moderate: 100-102°F
#   High: 101.5-104°F  |  Very High: > 103.5°F
# Overlap ensures smooth transitions and non-zero membership at boundaries.

def _fever_normal(x):
    return _trapezoidal(x, 93.0, 95.0, 97.5, 99.5)

def _fever_low_grade(x):
    return _triangular(x, 98.0, 99.5, 101.0)

def _fever_moderate(x):
    return _triangular(x, 99.5, 101.0, 102.5)

def _fever_high(x):
    return _triangular(x, 101.0, 102.5, 104.5)

def _fever_very_high(x):
    return _trapezoidal(x, 103.0, 104.5, 106.0, 108.0)

FEVER_FUZZY_SETS = {
    'normal':    _fever_normal,
    'low_grade': _fever_low_grade,
    'moderate':  _fever_moderate,
    'high':      _fever_high,
    'very_high': _fever_very_high,
}


# ─── Severity Fuzzy Sets (0-10 scale) ────────────────────────────────────────
# Redesigned with proper overlap so edge values get non-zero membership
# in adjacent sets. E.g. value=8 gets membership in both severe AND
# very_severe, which is clinically realistic.

def _severity_none(x):
    return _trapezoidal(x, -1.0, 0.0, 0.5, 2.5)

def _severity_mild(x):
    return _triangular(x, 0.5, 2.5, 5.0)

def _severity_moderate(x):
    return _triangular(x, 3.0, 5.0, 7.5)

def _severity_severe(x):
    return _triangular(x, 5.5, 7.5, 9.5)

def _severity_very_severe(x):
    return _trapezoidal(x, 7.5, 8.5, 10.0, 11.0)

SEVERITY_FUZZY_SETS = {
    'none':        _severity_none,
    'mild':        _severity_mild,
    'moderate':    _severity_moderate,
    'severe':      _severity_severe,
    'very_severe': _severity_very_severe,
}


# ─── Fuzzy Inference Engine ──────────────────────────────────────────────────

class TropicalFuzzySystem:
    """
    Fuzzy inference system for tropical disease prediction.

    For each disease, computes a weighted-average membership score by
    evaluating how well the patient's symptoms match the expected
    fuzzy pattern for that disease.
    """

    def get_membership(self, symptom_key, value):
        """
        Compute membership degrees for a symptom value across all fuzzy sets.

        Returns dict of {fuzzy_level: membership_degree}.
        """
        if symptom_key == 'fever':
            fuzzy_sets = FEVER_FUZZY_SETS
        else:
            fuzzy_sets = SEVERITY_FUZZY_SETS

        return {level: fn(float(value)) for level, fn in fuzzy_sets.items()}

    def predict(self, symptom_values):
        """
        Predict disease likelihoods from symptom values.

        Args:
            symptom_values: dict of {symptom_key: numeric_value}

        Returns:
            dict of {disease_name: confidence_score} (0.0 to 1.0),
            sorted by confidence descending.
        """
        if not symptom_values:
            return {}

        scores = {}

        for disease_name, rules in FUZZY_DISEASE_RULES.items():
            weighted_sum = 0.0
            total_weight = 0.0

            for symptom_key, (expected_level, weight) in rules.items():
                if symptom_key not in symptom_values:
                    continue

                value = float(symptom_values[symptom_key])

                # Get the membership degree for the expected fuzzy level
                if symptom_key == 'fever':
                    membership = FEVER_FUZZY_SETS[expected_level](value)
                else:
                    membership = SEVERITY_FUZZY_SETS[expected_level](value)

                weighted_sum += membership * weight
                total_weight += weight

            if total_weight > 0:
                scores[disease_name] = round(weighted_sum / total_weight, 4)

        # Sort by confidence descending
        return dict(sorted(scores.items(), key=lambda x: x[1], reverse=True))

    def get_detailed_analysis(self, symptom_values):
        """
        Return detailed per-symptom analysis for the top prediction.

        Useful for explaining WHY a particular disease was predicted.
        """
        scores = self.predict(symptom_values)
        if not scores:
            return None

        top_disease = list(scores.keys())[0]
        rules = FUZZY_DISEASE_RULES[top_disease]
        details = []

        for symptom_key, (expected_level, weight) in rules.items():
            if symptom_key not in symptom_values:
                continue

            value = float(symptom_values[symptom_key])
            all_memberships = self.get_membership(symptom_key, value)

            # Find which fuzzy set has highest membership for this value
            actual_level = max(all_memberships, key=all_memberships.get)
            match_degree = all_memberships.get(expected_level, 0)

            sym_info = SYMPTOMS[symptom_key]
            details.append({
                'symptom': sym_info['name'],
                'value': value,
                'expected_level': expected_level,
                'actual_level': actual_level,
                'match_degree': round(match_degree, 3),
                'weight': weight,
                'contribution': round(match_degree * weight, 3),
                'is_hallmark': symptom_key in DISEASE_PROFILES[top_disease].get('hallmarks', []),
            })

        details.sort(key=lambda x: x['contribution'], reverse=True)
        return {
            'disease': top_disease,
            'score': scores[top_disease],
            'details': details,
        }
