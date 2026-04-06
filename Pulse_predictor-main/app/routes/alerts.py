from fastapi import APIRouter, Request, Depends
from fastapi.responses import RedirectResponse, HTMLResponse
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import Alert, Project, User, Log
from app.auth import get_current_user
from app.services.alert_service import mark_alert_seen, mark_alert_acknowledged
from app.services.ai_service import deserialize_suggestions
from app.services.log_service import log_action
from app.templating import render

router = APIRouter(tags=["alerts"])


@router.get("/alerts", response_class=HTMLResponse)
async def list_alerts(
    request: Request,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    from sqlalchemy import func as sa_func
    import json as _json

    if user.role == "ADMIN":
        base_filter = True
    elif user.role == "MANAGER":
        base_filter = Alert.manager_id == user.id
    else:
        base_filter = True

    # Pagination
    page = int(request.query_params.get("page", 1))
    per_page = 50
    total_alerts = db.query(sa_func.count(Alert.alert_id)).filter(base_filter).scalar()
    total_pages = max(1, (total_alerts + per_page - 1) // per_page)
    page = max(1, min(page, total_pages))

    alerts = (
        db.query(Alert)
        .filter(base_filter)
        .order_by(Alert.created_at.desc())
        .offset((page - 1) * per_page)
        .limit(per_page)
        .all()
    )

    # Batch load project names (avoid N+1)
    project_ids = set(a.project_id for a in alerts)
    projects_map = {}
    if project_ids:
        proj_rows = db.query(Project.id, Project.name).filter(Project.id.in_(project_ids)).all()
        projects_map = {pid: pname for pid, pname in proj_rows}

    alert_data = []
    for a in alerts:
        suggestions = deserialize_suggestions(a.ai_suggestions) if a.ai_suggestions else {}
        alert_data.append({
            "alert": a,
            "project_name": projects_map.get(a.project_id, "Unknown"),
            "suggestions": suggestions,
        })

    msg = request.query_params.get("msg", "")

    # Chart data via SQL aggregation
    severity_rows = (
        db.query(Alert.severity, sa_func.count())
        .filter(base_filter)
        .group_by(Alert.severity)
        .all()
    )
    severity_counts = {"Safe": 0, "Warning": 0, "High Risk": 0}
    for sev, cnt in severity_rows:
        if sev in severity_counts:
            severity_counts[sev] = cnt

    status_rows = (
        db.query(Alert.status, sa_func.count())
        .filter(base_filter)
        .group_by(Alert.status)
        .all()
    )
    status_counts = {"UNREAD": 0, "SEEN": 0, "ACKNOWLEDGED": 0}
    for st, cnt in status_rows:
        if st in status_counts:
            status_counts[st] = cnt

    alert_chart = {
        "severity_labels": list(severity_counts.keys()),
        "severity_values": list(severity_counts.values()),
        "status_labels": list(status_counts.keys()),
        "status_values": list(status_counts.values()),
    }

    return render("alerts.html", request, {
        "user": user,
        "alert_data": alert_data,
        "total_alerts": total_alerts,
        "alert_chart_json": _json.dumps(alert_chart).replace("</", "<\\/"),
        "msg": msg,
        "page": page,
        "total_pages": total_pages,
    })


@router.post("/alerts/{alert_id}/seen")
async def alert_seen(
    alert_id: int,
    request: Request,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    alert = db.query(Alert).filter(Alert.alert_id == alert_id).first()
    if not alert:
        return RedirectResponse(url="/alerts?msg=Alert+not+found", status_code=303)

    # Ownership check
    if user.role != "ADMIN" and alert.manager_id != user.id:
        return RedirectResponse(url="/alerts?msg=Permission+denied", status_code=303)

    mark_alert_seen(db, alert_id, user.id)
    log_action(db, user.id, user.role, "ALERT_SEEN", f"/alerts/{alert_id}/seen", {"alert_id": alert_id})

    return RedirectResponse(url="/alerts?msg=Alert+marked+as+seen", status_code=303)


@router.post("/alerts/{alert_id}/acknowledge")
async def alert_acknowledge(
    alert_id: int,
    request: Request,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    alert = db.query(Alert).filter(Alert.alert_id == alert_id).first()
    if not alert:
        return RedirectResponse(url="/alerts?msg=Alert+not+found", status_code=303)

    if user.role != "ADMIN" and alert.manager_id != user.id:
        return RedirectResponse(url="/alerts?msg=Permission+denied", status_code=303)

    mark_alert_acknowledged(db, alert_id, user.id)
    log_action(
        db,
        user.id,
        user.role,
        "ALERT_ACKNOWLEDGED",
        f"/alerts/{alert_id}/acknowledge",
        {"alert_id": alert_id},
    )

    return RedirectResponse(url="/alerts?msg=Alert+acknowledged", status_code=303)


# ── Admin Panel ──
@router.get("/admin", response_class=HTMLResponse)
async def admin_panel(
    request: Request,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    if user.role != "ADMIN":
        return RedirectResponse(url="/projects?error=Admin+access+required", status_code=303)

    # Recent alerts (paginated, batch loaded)
    alerts = db.query(Alert).order_by(Alert.created_at.desc()).limit(100).all()

    # Batch load project names and manager names
    proj_ids = set(a.project_id for a in alerts)
    mgr_ids = set(a.manager_id for a in alerts)
    proj_map = {}
    mgr_map = {}
    if proj_ids:
        rows = db.query(Project.id, Project.name).filter(Project.id.in_(proj_ids)).all()
        proj_map = {pid: pname for pid, pname in rows}
    if mgr_ids:
        rows = db.query(User.id, User.name).filter(User.id.in_(mgr_ids)).all()
        mgr_map = {uid: uname for uid, uname in rows}

    alert_data = []
    for a in alerts:
        suggestions = deserialize_suggestions(a.ai_suggestions) if a.ai_suggestions else {}
        alert_data.append({
            "alert": a,
            "project_name": proj_map.get(a.project_id, "Unknown"),
            "manager_name": mgr_map.get(a.manager_id, "Unknown"),
            "suggestions": suggestions,
        })

    # Recent logs
    recent_logs = db.query(Log).order_by(Log.timestamp.desc()).limit(50).all()

    # Stats
    total_projects = db.query(Project).count()
    total_alerts = db.query(Alert).count()
    unread_alerts = db.query(Alert).filter(Alert.status == "UNREAD").count()
    total_users = db.query(User).count()

    msg = request.query_params.get("msg", "")

    return render("admin.html", request, {
        "user": user,
        "alert_data": alert_data,
        "recent_logs": recent_logs,
        "stats": {
            "total_projects": total_projects,
            "total_alerts": total_alerts,
            "unread_alerts": unread_alerts,
            "total_users": total_users,
        },
        "msg": msg,
    })
