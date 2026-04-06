import io
from fastapi import APIRouter, Request, Depends, Form, UploadFile, File
from fastapi.responses import RedirectResponse, HTMLResponse
from sqlalchemy.orm import Session
import pandas as pd

from app.database import get_db
from app.models import Project, Prediction, Alert, User
from app.auth import get_current_user
from app.services.project_service import verify_project_ownership, validate_project_data
from app.services.ml_service import predict
from app.services.alert_service import evaluate_and_create_alert
from app.services.log_service import log_action
from app.templating import render

router = APIRouter(tags=["projects"])


@router.get("/projects", response_class=HTMLResponse)
async def list_projects(
    request: Request,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    from sqlalchemy import func as sa_func
    import json as _json

    # ── Base query filter ──
    if user.role == "MANAGER":
        base_filter = Project.manager_id == user.id
    else:
        base_filter = True  # ADMIN and VIEWER see all

    msg = request.query_params.get("msg", "")
    error = request.query_params.get("error", "")

    # ── Pagination at DB level ──
    page = int(request.query_params.get("page", 1))
    per_page = 50
    total_projects = db.query(sa_func.count(Project.id)).filter(base_filter).scalar()
    total_pages = max(1, (total_projects + per_page - 1) // per_page)
    page = max(1, min(page, total_pages))

    projects = (
        db.query(Project)
        .filter(base_filter)
        .order_by(Project.id.desc())
        .offset((page - 1) * per_page)
        .limit(per_page)
        .all()
    )

    # Manager lookup for this page only
    manager_ids = set(p.manager_id for p in projects)
    managers_map = {}
    if manager_ids:
        mgr_users = db.query(User).filter(User.id.in_(manager_ids)).all()
        managers_map = {u.id: u.name for u in mgr_users}

    # Batch-load predictions for this page only
    page_project_ids = [p.id for p in projects]
    latest_pred_subq = (
        db.query(Prediction.project_id, sa_func.max(Prediction.id).label("max_id"))
        .filter(Prediction.project_id.in_(page_project_ids))
        .group_by(Prediction.project_id)
        .subquery()
    )
    latest_preds = (
        db.query(Prediction)
        .join(latest_pred_subq, Prediction.id == latest_pred_subq.c.max_id)
        .all()
    )
    pred_map = {pred.project_id: pred for pred in latest_preds}

    project_data = []
    for p in projects:
        project_data.append({
            "project": p,
            "prediction": pred_map.get(p.id),
            "manager_name": managers_map.get(p.manager_id, "Unassigned"),
        })

    # ── Chart data via SQL aggregation (fast, no full table load) ──

    # Risk distribution from predictions table
    risk_rows = (
        db.query(Prediction.predicted_risk, sa_func.count())
        .join(Project, Prediction.project_id == Project.id)
        .filter(base_filter)
        .filter(Prediction.predicted_risk.in_(["Safe", "Warning", "High Risk"]))
        .group_by(Prediction.predicted_risk)
        .all()
    )
    risk_counts = {"Safe": 0, "Warning": 0, "High Risk": 0}
    for label, cnt in risk_rows:
        if label in risk_counts:
            risk_counts[label] = cnt

    # Status distribution
    status_rows = (
        db.query(Project.status, sa_func.count())
        .filter(base_filter)
        .group_by(Project.status)
        .all()
    )
    status_counts = {(s or "Unknown"): c for s, c in status_rows}

    # Top 10 overruns via SQL (latest prediction per project)
    latest_pred_for_chart = (
        db.query(Prediction.project_id, sa_func.max(Prediction.id).label("max_id"))
        .group_by(Prediction.project_id)
        .subquery()
    )
    overrun_rows = (
        db.query(Project.name, Prediction.predicted_overrun)
        .join(latest_pred_for_chart, Project.id == latest_pred_for_chart.c.project_id)
        .join(Prediction, Prediction.id == latest_pred_for_chart.c.max_id)
        .filter(base_filter)
        .order_by(Prediction.predicted_overrun.desc())
        .limit(10)
        .all()
    )
    overrun_top = [{"name": n[:20], "overrun": round(o, 1)} for n, o in overrun_rows]

    # Top 12 cost via SQL
    cost_rows = (
        db.query(Project.name, Project.planned_cost, Project.actual_cost)
        .filter(base_filter)
        .order_by(Project.planned_cost.desc())
        .limit(12)
        .all()
    )
    cost_labels = [r[0][:18] for r in cost_rows]
    cost_planned = [round(r[1] or 0, 0) for r in cost_rows]
    cost_actual = [round(r[2] or 0, 0) for r in cost_rows]

    # Manager workload via SQL
    mgr_rows = (
        db.query(User.name, sa_func.count(Project.id))
        .join(Project, Project.manager_id == User.id)
        .filter(base_filter)
        .group_by(User.name)
        .all()
    )

    chart_data = {
        "risk_labels": list(risk_counts.keys()),
        "risk_values": list(risk_counts.values()),
        "status_labels": list(status_counts.keys()),
        "status_values": list(status_counts.values()),
        "overrun_labels": [o["name"] for o in overrun_top],
        "overrun_values": [o["overrun"] for o in overrun_top],
        "cost_labels": cost_labels,
        "cost_planned": cost_planned,
        "cost_actual": cost_actual,
        "manager_labels": [r[0] for r in mgr_rows],
        "manager_values": [r[1] for r in mgr_rows],
    }

    return render("dashboard.html", request, {
        "user": user,
        "project_data": project_data,
        "total_projects": total_projects,
        "risk_counts": risk_counts,
        "chart_data_json": _json.dumps(chart_data).replace("</", "<\\/"),
        "msg": msg,
        "error": error,
        "page": page,
        "total_pages": total_pages,
    })


@router.get("/projects/create", response_class=HTMLResponse)
async def create_project_page(
    request: Request, db: Session = Depends(get_db), user: User = Depends(get_current_user)
):
    if user.role not in ("ADMIN", "MANAGER"):
        return RedirectResponse(url="/projects?error=Permission+denied", status_code=303)
    managers = db.query(User).filter(User.role.in_(["ADMIN", "MANAGER"])).all() if user.role == "ADMIN" else []
    return render("create_project.html", request, {"user": user, "managers": managers})


@router.post("/projects/create")
async def create_project(
    request: Request,
    name: str = Form(...),
    planned_cost: float = Form(0),
    actual_cost: float = Form(0),
    planned_effort: float = Form(0),
    actual_effort: float = Form(0),
    resource_count: int = Form(1),
    start_date: str = Form(""),
    end_date: str = Form(""),
    tech_stack: str = Form(""),
    status: str = Form("Active"),
    manager_id: int = Form(0),
    mcc: str = Form(""),
    service_line: str = Form(""),
    segment: str = Form(""),
    service_offering: str = Form(""),
    contract_type: str = Form(""),
    baseline_rpp: float = Form(0),
    latest_rpp: float = Form(0),
    dollar_impact: float = Form(0),
    project_margin_baseline: float = Form(0),
    project_margin_latest: float = Form(0),
    onsite_mix_pct: float = Form(0),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    if user.role not in ("ADMIN", "MANAGER"):
        return RedirectResponse(url="/projects?error=Permission+denied", status_code=303)

    # Determine manager: ADMIN can assign, others default to self
    assigned_manager_id = user.id
    if user.role == "ADMIN" and manager_id > 0:
        assigned_manager_id = manager_id

    data = {
        "name": name,
        "planned_cost": planned_cost,
        "actual_cost": actual_cost,
        "planned_effort": planned_effort,
        "actual_effort": actual_effort,
        "resource_count": resource_count,
    }
    errors = validate_project_data(data)
    if errors:
        managers = db.query(User).filter(User.role.in_(["ADMIN", "MANAGER"])).all() if user.role == "ADMIN" else []
        return render("create_project.html", request, {"user": user, "managers": managers, "error": "; ".join(errors)})

    project = Project(
        name=name,
        manager_id=assigned_manager_id,
        planned_cost=planned_cost,
        actual_cost=actual_cost,
        planned_effort=planned_effort,
        actual_effort=actual_effort,
        resource_count=resource_count,
        start_date=start_date,
        end_date=end_date,
        tech_stack=tech_stack,
        status=status,
        mcc=mcc,
        service_line=service_line,
        segment=segment,
        service_offering=service_offering,
        contract_type=contract_type,
        baseline_rpp=baseline_rpp,
        latest_rpp=latest_rpp,
        dollar_impact=dollar_impact,
        project_margin_baseline=project_margin_baseline,
        project_margin_latest=project_margin_latest,
        onsite_mix_pct=onsite_mix_pct,
    )
    db.add(project)
    db.commit()
    db.refresh(project)

    # Run prediction pipeline
    prediction_result = predict(project)
    pred = Prediction(
        project_id=project.id,
        predicted_risk=prediction_result["risk"],
        predicted_overrun=prediction_result["overrun_pct"],
    )
    db.add(pred)
    db.commit()

    # Evaluate alerts
    evaluate_and_create_alert(db, project, prediction_result)

    log_action(
        db,
        user.id,
        user.role,
        "CREATE_PROJECT",
        "/projects/create",
        {"project_id": project.id, "project_name": name},
    )

    return RedirectResponse(url="/projects?msg=Project+created+successfully", status_code=303)


@router.get("/projects/upload", response_class=HTMLResponse)
async def upload_page(request: Request, user: User = Depends(get_current_user)):
    if user.role not in ("ADMIN", "MANAGER"):
        return RedirectResponse(url="/projects?error=Permission+denied", status_code=303)
    return render("upload.html", request, {"user": user})


@router.post("/projects/upload")
async def upload_csv(
    request: Request,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    if user.role not in ("ADMIN", "MANAGER"):
        return RedirectResponse(url="/projects?error=Permission+denied", status_code=303)

    if not file.filename.endswith(".csv"):
        return render("upload.html", request, {"user": user, "error": "Please upload a CSV file"})

    content = await file.read()
    if len(content) > 10 * 1024 * 1024:  # 10 MB limit
        return render("upload.html", request, {"user": user, "error": "File too large. Maximum size is 10 MB."})
    try:
        df = pd.read_csv(io.BytesIO(content))
    except Exception as e:
        return render("upload.html", request, {"user": user, "error": f"Error reading CSV: {str(e)}"})

    required_fields = ["name", "planned_cost", "actual_cost", "planned_effort", "actual_effort"]
    missing = [f for f in required_fields if f not in df.columns]
    if missing:
        return render("upload.html", request, {
            "user": user,
            "error": f"Missing required columns: {', '.join(missing)}",
        })

    created = 0
    errors_list = []

    for idx, row in df.iterrows():
        try:
            for field in ["planned_cost", "actual_cost", "planned_effort", "actual_effort"]:
                val = row.get(field)
                if pd.notna(val) and float(val) < 0:
                    errors_list.append(f"Row {idx + 1}: {field} cannot be negative")
                    break
            else:
                def safe_str(val, default=""):
                    if pd.isna(val):
                        return default
                    return str(val)

                # Resolve manager_id from CSV or default to uploader
                csv_manager_id = user.id
                if "manager_id" in row.index and pd.notna(row.get("manager_id")):
                    candidate = int(row["manager_id"])
                    mgr = db.query(User).filter(User.id == candidate, User.role.in_(["ADMIN", "MANAGER"])).first()
                    if mgr:
                        csv_manager_id = mgr.id

                project = Project(
                    name=str(row["name"]),
                    manager_id=csv_manager_id,
                    planned_cost=float(row.get("planned_cost", 0) or 0),
                    actual_cost=float(row.get("actual_cost", 0) or 0),
                    planned_effort=float(row.get("planned_effort", 0) or 0),
                    actual_effort=float(row.get("actual_effort", 0) or 0),
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
                    baseline_rpp=float(row.get("baseline_rpp", 0) or 0) if pd.notna(row.get("baseline_rpp")) else None,
                    latest_rpp=float(row.get("latest_rpp", 0) or 0) if pd.notna(row.get("latest_rpp")) else None,
                    dollar_impact=float(row.get("dollar_impact", 0) or 0) if pd.notna(row.get("dollar_impact")) else None,
                    project_margin_baseline=float(row.get("project_margin_baseline", 0) or 0) if pd.notna(row.get("project_margin_baseline")) else None,
                    project_margin_latest=float(row.get("project_margin_latest", 0) or 0) if pd.notna(row.get("project_margin_latest")) else None,
                    onsite_mix_pct=float(row.get("onsite_mix_pct", 0) or 0) if pd.notna(row.get("onsite_mix_pct")) else None,
                )
                db.add(project)
                db.commit()
                db.refresh(project)

                prediction_result = predict(project)
                pred = Prediction(
                    project_id=project.id,
                    predicted_risk=prediction_result["risk"],
                    predicted_overrun=prediction_result["overrun_pct"],
                )
                db.add(pred)
                db.commit()

                evaluate_and_create_alert(db, project, prediction_result)
                created += 1
        except Exception as e:
            db.rollback()
            errors_list.append(f"Row {idx + 1}: {str(e)}")

    log_action(
        db,
        user.id,
        user.role,
        "UPLOAD_CSV",
        "/projects/upload",
        {"projects_created": created, "errors": len(errors_list)},
    )

    msg = f"Successfully imported {created} projects"
    if errors_list:
        msg += f" with {len(errors_list)} errors"
    return RedirectResponse(url=f"/projects?msg={msg.replace(' ', '+')}", status_code=303)


@router.get("/projects/edit/{project_id}", response_class=HTMLResponse)
async def edit_project_page(
    project_id: int,
    request: Request,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    if user.role == "VIEWER":
        return RedirectResponse(url="/projects?error=Permission+denied", status_code=303)
    project = verify_project_ownership(db, project_id, user)
    managers = db.query(User).filter(User.role.in_(["ADMIN", "MANAGER"])).all() if user.role == "ADMIN" else []
    return render("edit_project.html", request, {"user": user, "project": project, "managers": managers})


@router.post("/projects/edit/{project_id}")
async def edit_project(
    project_id: int,
    request: Request,
    name: str = Form(...),
    planned_cost: float = Form(0),
    actual_cost: float = Form(0),
    planned_effort: float = Form(0),
    actual_effort: float = Form(0),
    resource_count: int = Form(1),
    start_date: str = Form(""),
    end_date: str = Form(""),
    tech_stack: str = Form(""),
    status: str = Form("Active"),
    manager_id: int = Form(0),
    mcc: str = Form(""),
    service_line: str = Form(""),
    segment: str = Form(""),
    service_offering: str = Form(""),
    contract_type: str = Form(""),
    baseline_rpp: float = Form(0),
    latest_rpp: float = Form(0),
    dollar_impact: float = Form(0),
    project_margin_baseline: float = Form(0),
    project_margin_latest: float = Form(0),
    onsite_mix_pct: float = Form(0),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    if user.role == "VIEWER":
        return RedirectResponse(url="/projects?error=Permission+denied", status_code=303)

    project = verify_project_ownership(db, project_id, user)

    data = {
        "name": name,
        "planned_cost": planned_cost,
        "actual_cost": actual_cost,
        "planned_effort": planned_effort,
        "actual_effort": actual_effort,
        "resource_count": resource_count,
    }
    errors = validate_project_data(data)
    if errors:
        managers = db.query(User).filter(User.role.in_(["ADMIN", "MANAGER"])).all() if user.role == "ADMIN" else []
        return render("edit_project.html", request, {"user": user, "project": project, "managers": managers, "error": "; ".join(errors)})

    # ADMIN can reassign manager
    if user.role == "ADMIN" and manager_id > 0:
        project.manager_id = manager_id

    project.name = name
    project.planned_cost = planned_cost
    project.actual_cost = actual_cost
    project.planned_effort = planned_effort
    project.actual_effort = actual_effort
    project.resource_count = resource_count
    project.start_date = start_date
    project.end_date = end_date
    project.tech_stack = tech_stack
    project.status = status
    project.mcc = mcc
    project.service_line = service_line
    project.segment = segment
    project.service_offering = service_offering
    project.contract_type = contract_type
    project.baseline_rpp = baseline_rpp
    project.latest_rpp = latest_rpp
    project.dollar_impact = dollar_impact
    project.project_margin_baseline = project_margin_baseline
    project.project_margin_latest = project_margin_latest
    project.onsite_mix_pct = onsite_mix_pct
    db.commit()
    db.refresh(project)

    # Re-run prediction pipeline
    prediction_result = predict(project)
    pred = Prediction(
        project_id=project.id,
        predicted_risk=prediction_result["risk"],
        predicted_overrun=prediction_result["overrun_pct"],
    )
    db.add(pred)
    db.commit()

    evaluate_and_create_alert(db, project, prediction_result)

    log_action(
        db,
        user.id,
        user.role,
        "UPDATE_PROJECT",
        f"/projects/edit/{project_id}",
        {"project_id": project.id},
    )

    return RedirectResponse(url="/projects?msg=Project+updated+successfully", status_code=303)


@router.post("/projects/delete/{project_id}")
async def delete_project(
    project_id: int,
    request: Request,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    if user.role not in ("ADMIN", "MANAGER"):
        return RedirectResponse(url="/projects?error=Permission+denied", status_code=303)

    project = verify_project_ownership(db, project_id, user)

    # Delete related records
    db.query(Prediction).filter(Prediction.project_id == project_id).delete()
    db.query(Alert).filter(Alert.project_id == project_id).delete()
    db.delete(project)
    db.commit()

    log_action(
        db,
        user.id,
        user.role,
        "DELETE_PROJECT",
        f"/projects/delete/{project_id}",
        {"project_id": project_id, "project_name": project.name},
    )

    return RedirectResponse(url="/projects?msg=Project+deleted", status_code=303)


# ── API Ingestion Endpoint ──
@router.post("/api/projects/ingest")
async def api_ingest(
    request: Request,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """API endpoint for JSON-based project ingestion."""
    if user.role not in ("ADMIN", "MANAGER"):
        from fastapi.responses import JSONResponse
        return JSONResponse(content={"error": "Permission denied"}, status_code=403)

    body = await request.json()
    projects_data = body if isinstance(body, list) else [body]

    results = []
    for item in projects_data:
        errors = validate_project_data(item)
        if errors:
            results.append({"name": item.get("name", "unknown"), "status": "error", "errors": errors})
            continue

        project = Project(
            name=item["name"],
            manager_id=user.id,
            planned_cost=item.get("planned_cost", 0),
            actual_cost=item.get("actual_cost", 0),
            planned_effort=item.get("planned_effort", 0),
            actual_effort=item.get("actual_effort", 0),
            resource_count=item.get("resource_count", 1),
            start_date=item.get("start_date", ""),
            end_date=item.get("end_date", ""),
            tech_stack=item.get("tech_stack", ""),
            status=item.get("status", "Active"),
            mcc=item.get("mcc", ""),
            service_line=item.get("service_line", ""),
            segment=item.get("segment", ""),
            service_offering=item.get("service_offering", ""),
            contract_type=item.get("contract_type", ""),
            baseline_rpp=item.get("baseline_rpp"),
            latest_rpp=item.get("latest_rpp"),
            dollar_impact=item.get("dollar_impact"),
            project_margin_baseline=item.get("project_margin_baseline"),
            project_margin_latest=item.get("project_margin_latest"),
            onsite_mix_pct=item.get("onsite_mix_pct"),
        )
        db.add(project)
        db.commit()
        db.refresh(project)

        prediction_result = predict(project)
        pred = Prediction(
            project_id=project.id,
            predicted_risk=prediction_result["risk"],
            predicted_overrun=prediction_result["overrun_pct"],
        )
        db.add(pred)
        db.commit()

        evaluate_and_create_alert(db, project, prediction_result)
        results.append({"name": project.name, "id": project.id, "status": "created", "risk": prediction_result["risk"]})

    log_action(db, user.id, user.role, "API_INGEST", "/api/projects/ingest", {"count": len(results)})

    return {"results": results}
