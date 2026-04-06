"""
Train ML models for Project Pulse Predictor.
Uses real overrun data from the enriched sample_data.csv (extracted from
Overrun_data_last_5_quarters.xlsb).  Trains a RandomForest classifier
(risk level) and a RandomForest regressor (cost overrun %), then saves
models to disk.

Usage:
    python train_model.py
"""

import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, mean_squared_error
import joblib

print("=" * 60)
print("  Project Pulse Predictor - Model Training (Real Data)")
print("=" * 60)

# ── Load Real Data ──
print("\n[1/5] Loading real project data from data/sample_data.csv...")

csv_path = os.path.join(os.path.dirname(__file__), "data", "sample_data.csv")
df = pd.read_csv(csv_path)
print(f"   Loaded {len(df)} projects")

# ── Feature Engineering ──
print("[2/5] Computing engineered features...")

planned_cost = pd.to_numeric(df["planned_cost"], errors="coerce").fillna(0)
actual_cost = pd.to_numeric(df["actual_cost"], errors="coerce").fillna(0)
planned_effort = pd.to_numeric(df["planned_effort"], errors="coerce").fillna(0)
actual_effort = pd.to_numeric(df["actual_effort"], errors="coerce").fillna(0)
resource_count = pd.to_numeric(df["resource_count"], errors="coerce").fillna(1).clip(lower=1)

# Duration from start_date / end_date
start_dates = pd.to_datetime(df["start_date"], errors="coerce")
end_dates = pd.to_datetime(df["end_date"], errors="coerce")
duration = (end_dates - start_dates).dt.days.fillna(1).clip(lower=1)

cost_variance = (actual_cost - planned_cost) / planned_cost.clip(lower=1)
effort_variance = (actual_effort - planned_effort) / planned_effort.clip(lower=1)
burn_rate = actual_cost / duration.clip(lower=1)
resource_utilization = actual_effort / (resource_count * duration / 30).clip(lower=1)

# RPP delta
baseline_rpp = pd.to_numeric(df.get("baseline_rpp", 0), errors="coerce").fillna(0)
latest_rpp = pd.to_numeric(df.get("latest_rpp", 0), errors="coerce").fillna(0)
rpp_delta = (latest_rpp - baseline_rpp) / baseline_rpp.clip(lower=1)
rpp_delta = rpp_delta.replace([np.inf, -np.inf], 0).fillna(0)

# Margin delta
margin_baseline = pd.to_numeric(df.get("project_margin_baseline", 0), errors="coerce").fillna(0)
margin_latest = pd.to_numeric(df.get("project_margin_latest", 0), errors="coerce").fillna(0)
margin_delta = margin_latest - margin_baseline

# Onsite mix
onsite_mix = pd.to_numeric(df.get("onsite_mix_pct", 0), errors="coerce").fillna(0)

features = pd.DataFrame({
    "cost_variance": cost_variance,
    "effort_variance": effort_variance,
    "burn_rate": burn_rate,
    "resource_utilization": resource_utilization,
    "duration": duration,
    "resource_count": resource_count,
    "rpp_delta": rpp_delta,
    "margin_delta": margin_delta,
    "onsite_mix": onsite_mix,
})

# Replace infinities
features = features.replace([np.inf, -np.inf], 0).fillna(0)

# ── Labels from real data ──
# Risk labels: derive from cost variance + effort variance + dollar_impact
dollar_impact = pd.to_numeric(df.get("dollar_impact", 0), errors="coerce").fillna(0)

# Use the real Overrun/Underrun status + magnitude for risk classification
risk_score = (
    np.abs(cost_variance) * 0.35
    + np.abs(effort_variance) * 0.25
    + (np.abs(rpp_delta) > 0.1).astype(float) * 0.15
    + (margin_delta < -0.05).astype(float) * 0.15
    + (resource_utilization > 1.0).astype(float) * 0.10
)

risk_labels = np.where(
    risk_score > 0.25, "High Risk",
    np.where(risk_score > 0.08, "Warning", "Safe")
)

# Cost overrun percentage (clipped to avoid extreme outliers)
cost_overrun_pct = (cost_variance * 100).clip(-200, 500)

print(f"   Features: {list(features.columns)}")
print(f"   Safe: {(risk_labels == 'Safe').sum()}, Warning: {(risk_labels == 'Warning').sum()}, High Risk: {(risk_labels == 'High Risk').sum()}")

# ── Train/Test Split ──
print("[3/5] Splitting data (80/20)...")

X_train, X_test, y_cls_train, y_cls_test = train_test_split(
    features, risk_labels, test_size=0.2, random_state=42, stratify=risk_labels
)
_, _, y_reg_train, y_reg_test = train_test_split(
    features, cost_overrun_pct, test_size=0.2, random_state=42, stratify=risk_labels
)

# ── Train Models ──
print("[4/5] Training models...")

clf = RandomForestClassifier(n_estimators=200, max_depth=12, random_state=42)
clf.fit(X_train, y_cls_train)

reg = RandomForestRegressor(n_estimators=200, max_depth=12, random_state=42)
reg.fit(X_train, y_reg_train)

# ── Evaluate ──
print("\n--- Classification Report ---")
print(classification_report(y_cls_test, clf.predict(X_test)))

reg_pred = reg.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_reg_test, reg_pred))
print(f"--- Regression RMSE: {rmse:.2f}% ---\n")

# Feature importance
print("Feature Importance (Classifier):")
for feat, imp in sorted(zip(features.columns, clf.feature_importances_), key=lambda x: -x[1]):
    print(f"   {feat}: {imp:.4f}")

# ── Save Models ──
print("\n[5/5] Saving models...")

os.makedirs("models", exist_ok=True)
joblib.dump(clf, "models/risk_classifier.joblib")
joblib.dump(reg, "models/overrun_regressor.joblib")

print("   Saved: models/risk_classifier.joblib")
print("   Saved: models/overrun_regressor.joblib")
print("\n" + "=" * 60)
print("  Training complete! You can now run the application.")
print("=" * 60)
