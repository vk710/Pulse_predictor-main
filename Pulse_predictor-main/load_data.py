"""Load all 5 quarters of project data directly into PPP database (fast bulk insert).

Uses vectorized ML predictions via numpy/pandas for speed (~2 min instead of 30+).
"""
import os
import sys
import numpy as np
import bcrypt
import pandas as pd
from datetime import datetime, timedelta

# Ensure the project root is on the path
sys.path.insert(0, os.path.dirname(__file__))

from app.database import engine, SessionLocal, Base
from app.models import User, Project, Prediction, Alert
from app.config import COST_VARIANCE_THRESHOLD, EFFORT_VARIANCE_THRESHOLD, RISK_SCORE_THRESHOLD
import joblib

# ── Create tables ──
Base.metadata.create_all(bind=engine)
db = SessionLocal()

# ── Register demo users ──
users = [
    ("Admin User", "admin@test.com", "admin123", "ADMIN"),
    ("Sarah Johnson", "manager@test.com", "manager123", "MANAGER"),
    ("Viewer User", "viewer@test.com", "viewer123", "VIEWER"),
    ("Mike Chen", "mike@test.com", "manager123", "MANAGER"),
    ("Priya Patel", "priya@test.com", "manager123", "MANAGER"),
]
manager_ids = []
for name, email, pwd, role in users:
    existing = db.query(User).filter(User.email == email).first()
    if existing:
        print(f"  Already exists: {email}")
        if role in ("ADMIN", "MANAGER"):
            manager_ids.append(existing.id)
        continue
    hashed = bcrypt.hashpw(pwd.encode(), bcrypt.gensalt()).decode()
    u = User(name=name, email=email, password_hash=hashed, role=role)
    db.add(u)
    db.commit()
    db.refresh(u)
    print(f"  Registered {role}: {email} (id={u.id})")
    if role in ("ADMIN", "MANAGER"):
        manager_ids.append(u.id)

if not manager_ids:
    manager_ids = [u.id for u in db.query(User).filter(User.role.in_(["ADMIN", "MANAGER"])).all()]

print(f"\nManager IDs: {manager_ids}")

# ── Load CSV ──
data_dir = os.path.join(os.path.dirname(__file__), "data")
csv_path = os.path.join(data_dir, "training_data.csv")
if not os.path.exists(csv_path):
    csv_path = os.path.join(data_dir, "sample_data.csv")
    print(f"training_data.csv not found, falling back to sample_data.csv")

df = pd.read_csv(csv_path)
print(f"Loaded {len(df)} rows from {os.path.basename(csv_path)}")
if "quarter" in df.columns:
    print(f"  Quarters: {df['quarter'].value_counts().to_dict()}")

# ── Load ML models once ──
model_dir = os.path.join(os.path.dirname(__file__), "models")
classifier = joblib.load(os.path.join(model_dir, "risk_classifier.joblib"))
regressor = joblib.load(os.path.join(model_dir, "overrun_regressor.joblib"))
print("ML models loaded")

# ── Compute all features vectorized (pandas) ──
print("Computing features for all rows...")


def parse_date_col(col):
    return pd.to_datetime(col, errors="coerce")


start_dates = parse_date_col(df["start_date"])
end_dates = parse_date_col(df["end_date"])
duration = ((end_dates - start_dates).dt.days).clip(lower=1).fillna(1).astype(float)

planned_cost = df["planned_cost"].fillna(0).astype(float)
actual_cost = df["actual_cost"].fillna(0).astype(float)
planned_effort = df["planned_effort"].fillna(0).astype(float)
actual_effort = df["actual_effort"].fillna(0).astype(float)
resource_count = df["resource_count"].fillna(1).astype(float)

cost_variance = (actual_cost - planned_cost) / planned_cost.clip(lower=1)
effort_variance = (actual_effort - planned_effort) / planned_effort.clip(lower=1)
burn_rate = actual_cost / duration.clip(lower=1)
resource_utilization = actual_effort / (resource_count * duration / 30).clip(lower=1)

baseline_rpp = df["baseline_rpp"].fillna(0).astype(float) if "baseline_rpp" in df.columns else pd.Series(0, index=df.index)
latest_rpp = df["latest_rpp"].fillna(0).astype(float) if "latest_rpp" in df.columns else pd.Series(0, index=df.index)
rpp_delta = np.where(baseline_rpp > 0, (latest_rpp - baseline_rpp) / baseline_rpp.clip(lower=1), 0)

margin_baseline = df["project_margin_baseline"].fillna(0).astype(float) if "project_margin_baseline" in df.columns else pd.Series(0, index=df.index)
margin_latest = df["project_margin_latest"].fillna(0).astype(float) if "project_margin_latest" in df.columns else pd.Series(0, index=df.index)
margin_delta = margin_latest - margin_baseline

onsite_mix = df["onsite_mix_pct"].fillna(0).astype(float) if "onsite_mix_pct" in df.columns else pd.Series(0, index=df.index)

features_df = pd.DataFrame({
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

# ── Vectorized ML predictions (entire dataset at once) ──
print("Running ML predictions on all rows...")
risk_labels = classifier.predict(features_df)
risk_proba = classifier.predict_proba(features_df)
classes = list(classifier.classes_)
hr_idx = classes.index("High Risk") if "High Risk" in classes else 0
risk_scores = risk_proba[:, hr_idx]
overrun_pcts = regressor.predict(features_df)

print(f"  Risk distribution: {pd.Series(risk_labels).value_counts().to_dict()}")

# ── Bulk insert projects + predictions + alerts ──
BATCH_SIZE = 2000
total = len(df)

print(f"\nInserting {total} projects in batches of {BATCH_SIZE}...")

for batch_start in range(0, total, BATCH_SIZE):
    batch_end = min(batch_start + BATCH_SIZE, total)
    batch_df = df.iloc[batch_start:batch_end]
    projects = []

    for i, (_, row) in enumerate(batch_df.iterrows()):
        def safe_float(val, default=None):
            if pd.isna(val):
                return default
            return float(val)

        def safe_str(val, default=""):
            if pd.isna(val):
                return default
            return str(val)

        global_idx = batch_start + i
        mgr_id = manager_ids[global_idx % len(manager_ids)]

        p = Project(
            name=str(row["name"]),
            manager_id=mgr_id,
            planned_cost=safe_float(row.get("planned_cost"), 0),
            actual_cost=safe_float(row.get("actual_cost"), 0),
            planned_effort=safe_float(row.get("planned_effort"), 0),
            actual_effort=safe_float(row.get("actual_effort"), 0),
            resource_count=int(row.get("resource_count", 1) if pd.notna(row.get("resource_count")) else 1),
            start_date=safe_str(row.get("start_date")),
            end_date=safe_str(row.get("end_date")),
            tech_stack=safe_str(row.get("tech_stack")),
            status=safe_str(row.get("status"), "Active"),
            mcc=safe_str(row.get("mcc")),
            service_line=safe_str(row.get("service_line")),
            segment=safe_str(row.get("segment")),
            service_offering=safe_str(row.get("service_offering")),
            contract_type=safe_str(row.get("contract_type")),
            baseline_rpp=safe_float(row.get("baseline_rpp")),
            latest_rpp=safe_float(row.get("latest_rpp")),
            dollar_impact=safe_float(row.get("dollar_impact")),
            project_margin_baseline=safe_float(row.get("project_margin_baseline")),
            project_margin_latest=safe_float(row.get("project_margin_latest")),
            onsite_mix_pct=safe_float(row.get("onsite_mix_pct")),
            quarter=safe_str(row.get("quarter")) if "quarter" in row.index else None,
        )
        projects.append(p)

    db.add_all(projects)
    db.flush()  # get IDs assigned

    # Add pre-computed predictions
    for i, p in enumerate(projects):
        global_idx = batch_start + i
        pred = Prediction(
            project_id=p.id,
            predicted_risk=str(risk_labels[global_idx]),
            predicted_overrun=round(float(overrun_pcts[global_idx]), 2),
        )
        db.add(pred)

        # Create alerts where thresholds exceeded
        cv = float(cost_variance.iloc[global_idx])
        ev = float(effort_variance.iloc[global_idx])
        rs = float(risk_scores[global_idx])

        if cv > COST_VARIANCE_THRESHOLD:
            severity = "High Risk" if cv > 0.3 else ("Warning" if cv > 0.2 else "Low")
            db.add(Alert(
                project_id=p.id, manager_id=p.manager_id,
                alert_type="COST_OVERRUN", severity=severity,
                risk_score=round(rs, 4),
                message=f"Cost variance of {cv:.1%} exceeds threshold of {COST_VARIANCE_THRESHOLD:.1%}",
            ))
        if ev > EFFORT_VARIANCE_THRESHOLD:
            severity = "High Risk" if ev > 0.3 else ("Warning" if ev > 0.2 else "Low")
            db.add(Alert(
                project_id=p.id, manager_id=p.manager_id,
                alert_type="EFFORT_OVERRUN", severity=severity,
                risk_score=round(rs, 4),
                message=f"Effort variance of {ev:.1%} exceeds threshold of {EFFORT_VARIANCE_THRESHOLD:.1%}",
            ))
        if rs > RISK_SCORE_THRESHOLD:
            db.add(Alert(
                project_id=p.id, manager_id=p.manager_id,
                alert_type="HIGH_RISK", severity="High Risk",
                risk_score=round(rs, 4),
                message=f"Risk score of {rs:.2f} exceeds threshold of {RISK_SCORE_THRESHOLD}",
            ))

    db.commit()
    pct = min(100, round(100 * batch_end / total))
    print(f"  {batch_end:>6}/{total} ({pct}%)")

# ── Summary ──
total_projects = db.query(Project).count()
total_predictions = db.query(Prediction).count()
total_alerts = db.query(Alert).count()

print(f"\n=== Done ===")
print(f"  Projects:    {total_projects}")
print(f"  Predictions: {total_predictions}")
print(f"  Alerts:      {total_alerts}")
print(f"\nVisit http://127.0.0.1:8000")
print(f"Login: admin@test.com / admin123")

db.close()
