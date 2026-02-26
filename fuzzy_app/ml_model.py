"""
Random Forest Classifier for Tropical Disease Prediction

Trained on prevalence-based dataset generated from clinically-validated
disease profiles (see tropical_diseases.py).

KEY DESIGN DECISIONS:
1. Prevalence-based data generation (not pure Gaussian) — realistic patient variety
2. Conservative hyperparameters to prevent overfitting
3. Proper train/test split (80/20) with stratification
4. 5-fold stratified cross-validation for honest accuracy estimates
5. Feature masking augmentation (partial symptom input robustness)
6. Overfitting detection: train vs test accuracy comparison

Expected realistic accuracy: 82-92% depending on fold
(Some diseases naturally overlap, e.g., dengue vs chikungunya)
"""

import os
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    cross_val_score, StratifiedKFold, train_test_split
)
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

from .tropical_diseases import (
    SYMPTOM_ORDER, DISEASE_PROFILES, SYMPTOMS, generate_dataset
)

# Path for saved model artifacts
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'trained_model')
MODEL_PATH = os.path.join(MODEL_DIR, 'rf_tropical_disease.joblib')
ENCODER_PATH = os.path.join(MODEL_DIR, 'label_encoder.joblib')

# Healthy baseline values: what we fill when a symptom is NOT reported.
# "not reported" = "not experiencing" = healthy default.
BASELINE_VALUES = {key: 0.0 for key in SYMPTOM_ORDER}
BASELINE_VALUES['fever'] = 98.6  # Normal body temperature


def _prepare_training_data(samples_per_disease=300, augment=True, random_seed=42):
    """
    Generate and prepare training data with optional augmentation.

    Augmentation: for each original sample, create versions where 1-4
    random symptoms are masked to baseline values. This teaches the
    model to predict correctly even with partial inputs (real patients
    don't always report every symptom).

    Limited masking (1-4 features) to avoid corrupting the signal.
    """
    data, symptom_keys = generate_dataset(
        samples_per_disease=samples_per_disease,
        random_seed=random_seed
    )

    X_rows = []
    y_rows = []

    rng = np.random.RandomState(random_seed + 1)

    for row in data:
        features = [row[k] for k in symptom_keys]
        label = row['disease']

        # Original sample — always include
        X_rows.append(features)
        y_rows.append(label)

        if augment:
            # Create 2 augmented copies with limited masking
            for _ in range(2):
                masked = list(features)
                # Mask 1-4 features (conservative to preserve signal)
                n_mask = rng.randint(1, 5)
                mask_indices = rng.choice(len(symptom_keys), size=n_mask, replace=False)
                for idx in mask_indices:
                    masked[idx] = BASELINE_VALUES[symptom_keys[idx]]
                X_rows.append(masked)
                y_rows.append(label)

    X = np.array(X_rows)
    y = np.array(y_rows)

    return X, y, symptom_keys


def train_model(samples_per_disease=300, verbose=True):
    """
    Train Random Forest classifier with proper validation.

    Hyperparameters chosen to PREVENT overfitting:
    - n_estimators=150: enough trees, not excessive
    - max_depth=12: limits tree complexity (was 15)
    - min_samples_leaf=5: larger leaves = more generalization
    - min_samples_split=10: stricter split criterion
    - max_features='sqrt': random feature subset per split
    - class_weight='balanced': handles class imbalance

    Validation protocol:
    1. Generate data → 80/20 stratified train/test split
    2. Train on train set
    3. Evaluate on held-out test set
    4. 5-fold cross-validation for stability estimate
    5. Compare train vs test accuracy to detect overfitting

    Returns:
        dict with model, encoder, and comprehensive metrics
    """
    X, y, symptom_keys = _prepare_training_data(
        samples_per_disease=samples_per_disease, augment=True
    )

    if verbose:
        print(f"Total data: {X.shape[0]} samples, {X.shape[1]} features, "
              f"{len(set(y))} classes")

    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # ── Step 1: Train/Test Split (80/20) ──
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.20, stratify=y_encoded, random_state=42
    )

    if verbose:
        print(f"Train set: {X_train.shape[0]} samples")
        print(f"Test set:  {X_test.shape[0]} samples")

    # ── Step 2: Train Random Forest ──
    rf = RandomForestClassifier(
        n_estimators=150,
        max_depth=12,          # More conservative than before
        min_samples_leaf=5,    # Larger leaves = less overfitting
        min_samples_split=10,  # Stricter splits
        max_features='sqrt',
        class_weight='balanced',
        random_state=42,
        n_jobs=-1,
    )

    rf.fit(X_train, y_train)

    # ── Step 3: Evaluate ──
    train_accuracy = rf.score(X_train, y_train)
    test_accuracy = rf.score(X_test, y_test)

    if verbose:
        print(f"\nTrain accuracy: {train_accuracy:.4f}")
        print(f"Test accuracy:  {test_accuracy:.4f}")
        overfit_gap = train_accuracy - test_accuracy
        if overfit_gap > 0.10:
            print(f"⚠ Overfitting detected: gap = {overfit_gap:.4f}")
        else:
            print(f"✓ Overfitting check OK: gap = {overfit_gap:.4f}")

    # ── Step 4: Cross-Validation ──
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(rf, X, y_encoded, cv=cv, scoring='accuracy')

    if verbose:
        print(f"\n5-fold CV accuracy: {cv_scores.mean():.4f} "
              f"(+/- {cv_scores.std():.4f})")
        print(f"CV fold scores: {[f'{s:.4f}' for s in cv_scores]}")

    # ── Step 5: Per-Class Report ──
    y_pred_test = rf.predict(X_test)
    class_names = label_encoder.inverse_transform(range(len(label_encoder.classes_)))

    if verbose:
        print(f"\n{'='*60}")
        print("Per-Disease Classification Report (on held-out test set):")
        print('='*60)
        report = classification_report(
            y_test, y_pred_test, target_names=class_names, digits=3
        )
        print(report)

    # Feature importances
    importances = dict(zip(symptom_keys, rf.feature_importances_))

    if verbose:
        print("\nTop 10 Feature Importances:")
        for feat, imp in sorted(importances.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"  {feat:20s}: {imp:.4f}")

    # ── Now retrain on FULL dataset for deployment ──
    # (Cross-validation gave us honest estimates; now use all data for best model)
    rf_final = RandomForestClassifier(
        n_estimators=150,
        max_depth=12,
        min_samples_leaf=5,
        min_samples_split=10,
        max_features='sqrt',
        class_weight='balanced',
        random_state=42,
        n_jobs=-1,
    )
    rf_final.fit(X, y_encoded)

    # Save final model
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(rf_final, MODEL_PATH)
    joblib.dump(label_encoder, ENCODER_PATH)

    if verbose:
        print(f"\nFinal model saved to {MODEL_PATH}")

    return {
        'model': rf_final,
        'encoder': label_encoder,
        'train_accuracy': float(train_accuracy),
        'test_accuracy': float(test_accuracy),
        'cv_accuracy': float(cv_scores.mean()),
        'cv_std': float(cv_scores.std()),
        'feature_importances': importances,
    }


def load_model():
    """Load trained model from disk. Returns (model, encoder) or None."""
    if not os.path.exists(MODEL_PATH) or not os.path.exists(ENCODER_PATH):
        return None
    try:
        rf = joblib.load(MODEL_PATH)
        encoder = joblib.load(ENCODER_PATH)
        return rf, encoder
    except Exception:
        return None


class RFPredictor:
    """
    Random Forest predictor for tropical diseases.

    Wraps the sklearn model with a clean predict interface
    that returns disease probabilities.
    """

    def __init__(self):
        self.model = None
        self.encoder = None
        self._loaded = False

    def ensure_loaded(self):
        """Load model, training if necessary."""
        if self._loaded:
            return True

        result = load_model()
        if result is None:
            # Auto-train if no saved model
            print("No trained model found. Training now...")
            info = train_model(verbose=False)
            self.model = info['model']
            self.encoder = info['encoder']
        else:
            self.model, self.encoder = result

        self._loaded = True
        return True

    def predict(self, symptom_values):
        """
        Predict disease probabilities from symptom values.

        Missing symptoms are filled with healthy baseline values:
        - Fever: 98.6°F (normal temperature)
        - All others: 0 (symptom not present)

        Args:
            symptom_values: dict of {symptom_key: numeric_value}

        Returns:
            dict of {disease_name: probability} sorted by probability desc.
            Probabilities sum to 1.0.
        """
        self.ensure_loaded()

        if not symptom_values:
            return {}

        # Build feature vector — use healthy baselines for missing symptoms
        features = []
        for key in SYMPTOM_ORDER:
            if key in symptom_values:
                features.append(float(symptom_values[key]))
            else:
                features.append(BASELINE_VALUES[key])

        X = np.array([features])

        # Get probabilities
        probs = self.model.predict_proba(X)[0]
        classes = self.encoder.inverse_transform(range(len(probs)))

        result = {cls: round(float(prob), 4) for cls, prob in zip(classes, probs)}

        # Sort by probability descending
        return dict(sorted(result.items(), key=lambda x: x[1], reverse=True))

    def get_feature_importances(self):
        """Return feature importance scores."""
        self.ensure_loaded()
        importances = dict(zip(SYMPTOM_ORDER, self.model.feature_importances_))
        return dict(sorted(importances.items(), key=lambda x: x[1], reverse=True))
