import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings("ignore")

# -------------------------------
# STEP 1: Load dataset
# -------------------------------
data = pd.read_csv("sleep_apnea_dataset.csv")

# Check for missing values
data = data.fillna(0)

# -------------------------------
# STEP 2: Feature Engineering (Time-Series → Per Patient)
# -------------------------------
def extract_features(df):
    features = df.groupby('patient_id').agg({
        'heart_rate': ['mean', 'std', 'min', 'max', 'median'],
        'spo2': ['mean', 'std', 'min', 'max', 'median'],
        'hr_mean': ['mean'],
        'spo2_mean': ['mean'],
        'hr_std': ['mean'],
        'spo2_std': ['mean'],
        'hr_diff': ['mean', 'std'],
        'spo2_diff': ['mean', 'std'],
        'apnea_label': ['mean', 'max']
    })
    features.columns = ['_'.join(col) for col in features.columns]
    features = features.reset_index()
    
    # Derived features
    features['hr_spo2_ratio_mean'] = features['heart_rate_mean'] / (features['spo2_mean'] + 1e-6)
    features['spo2_drop_range'] = features['spo2_max'] - features['spo2_min']
    features['hr_fluctuation'] = features['heart_rate_max'] - features['heart_rate_min']
    features['variability_score'] = features['heart_rate_std'] + features['spo2_std']
    features['apnea_events'] = features['apnea_label_sum'] = features['apnea_label_max']  # 1 if any apnea event
    
    return features

patient_features = extract_features(data)

# -------------------------------
# STEP 3: Prepare Data for Training
# -------------------------------
X = patient_features.drop(columns=['patient_id', 'apnea_label_max'])
y = patient_features['apnea_label_max']

# Train-Test Split (Stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42
)

# -------------------------------
# STEP 4: Model Training (XGBoost)
# -------------------------------
params = {
    'max_depth': [3, 4, 5],
    'learning_rate': [0.05, 0.1, 0.2],
    'n_estimators': [100, 200],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0],
    'scale_pos_weight': [1, len(y[y == 0]) / len(y[y == 1])]
}

xgb = XGBClassifier(eval_metric='logloss', random_state=42)
grid = GridSearchCV(xgb, params, scoring='f1', cv=5, n_jobs=-1)
grid.fit(X_train, y_train)

best_model = grid.best_estimator_

# -------------------------------
# STEP 5: Model Evaluation
# -------------------------------
y_prob = best_model.predict_proba(X_test)[:, 1]
y_pred = (y_prob > 0.35).astype(int)  # Lower threshold to improve apnea recall

print("✅ Best Parameters:", grid.best_params_)

cm = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred, digits=3)
roc_auc = roc_auc_score(y_test, y_prob)
acc = accuracy_score(y_test, y_pred)

print(f"\n🎯 Accuracy: {acc:.3f}")
print(f"ROC-AUC Score: {roc_auc:.3f}")
print("\nConfusion Matrix:\n", cm)
print("\nClassification Report:\n", report)

# -------------------------------
# STEP 6: Save Model
# -------------------------------
joblib.dump(best_model, "improved_sleep_apnea_model.pkl")
print("💾 Model saved as improved_sleep_apnea_model.pkl")
