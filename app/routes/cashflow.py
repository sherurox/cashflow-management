# app/routes/cashflow.py

from app.services.knowledge_package import upsert_knowledge_package
from app.services.pinecone_index import publish_to_pinecone

from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from bson import Binary
import shortuuid
import pandas as pd
import logging
from decimal import Decimal, ROUND_HALF_UP
from typing import Optional

from app.db.mongo import get_db
from app.utils.parsing import load_and_normalize_csv

from app.services.aggregation import (
    compute_weekly_history,
    schedule_paid_outflows_with_payables,
    baseline_13_week_forecast_from_history,
    compute_weekly_by_category,
    baseline_13_week_category_forecast,
    compute_weekly_by_vendor_outflows,
    top_vendors_by_total_outflow,
    baseline_13_week_vendor_forecast,
    detect_recurring_payments,
    project_recurring_to_weeks,

    # ---- QA outputs (Part 2) ----
    historical_min_balance_and_date,
    forecast_min_balance_and_week,
    forecast_outflow_anomaly_warning,
)

router = APIRouter()
logger = logging.getLogger("cashflow")


# -------------------------
# Helpers
# -------------------------

def _get_totals_doc(db, ingestion_id: str, expected_starting_balance: float):
    """
    Always use totals forecast stored under type='totals' IF it matches the current starting balance.
    If missing or mismatch, build it by calling forecast(ingestion_id) once, then fetch again.
    """
    doc = db.forecasts.find_one({"ingestion_id": ingestion_id, "type": "totals"}, {"_id": 0})
    if doc and doc.get("weeks"):
        used = doc.get("starting_balance_used", None)
        try:
            used_f = float(used) if used is not None else None
        except Exception:
            used_f = None

        if used_f is not None and abs(used_f - float(expected_starting_balance)) < 1e-9:
            return doc

    # build totals once (upserts totals)
    _ = forecast(ingestion_id)

    doc = db.forecasts.find_one({"ingestion_id": ingestion_id, "type": "totals"}, {"_id": 0})
    if not doc or not doc.get("weeks"):
        raise HTTPException(status_code=500, detail="Totals not built")
    return doc


def _get_txns_or_404(db, ingestion_id: str):
    txns = list(db.transactions.find({"ingestion_id": ingestion_id}, {"_id": 0}))
    if not txns:
        raise HTTPException(status_code=404, detail="No transactions found for ingestion_id")
    return txns


def _week_key(x) -> pd.Timestamp:
    """Canonicalize any date-like value to a comparable Timestamp (date only)."""
    return pd.to_datetime(x).normalize()


def _cat_key(s: str) -> str:
    """Canonicalize category strings for stable joining."""
    return str(s).strip().lower()


def _get_horizon_weeks(txns):
    """Return the canonical 13-week horizon week_start dates.

    We anchor the horizon to the last observed historical week_start (from weekly_history)
    and then generate the next 13 weekly periods. This avoids relying on any baseline
    forecaster output to define the horizon and keeps the schedule deterministic.
    """
    weekly_hist = compute_weekly_history(txns)
    if weekly_hist is None or getattr(weekly_hist, "empty", True) or "week_start" not in weekly_hist.columns:
        raise HTTPException(status_code=400, detail="Insufficient history to build horizon weeks")

    last_ws = _week_key(weekly_hist["week_start"].max())
    # Next week starts 7 days after the last historical week_start
    start = last_ws + pd.Timedelta(days=7)
    horizon = pd.date_range(start=start, periods=13, freq="W-MON")

    # If the historical week_start is not Monday-normalized, fallback to simple 7D steps
    # to preserve continuity.
    if len(horizon) != 13:
        horizon = [start + pd.Timedelta(days=7 * i) for i in range(13)]

    return [_week_key(x) for x in list(horizon)]


def _build_recurring_maps(txns, horizon_weeks):
    patterns = detect_recurring_payments(txns)
    recurring_proj = project_recurring_to_weeks(patterns, horizon_weeks)

    rec_outflow_by_week = {}
    rec_outflow_by_week_cat = {}

    if recurring_proj is not None and not getattr(recurring_proj, "empty", True):
        rp = recurring_proj.copy()
        rp["week_start"] = rp["week_start"].apply(_week_key)

        if "outflow_total" in rp.columns:
            rec_outflow_by_week = rp.groupby("week_start")["outflow_total"].sum().to_dict()

        cat_col = "category_normalized" if "category_normalized" in rp.columns else "category"
        if cat_col in rp.columns and "outflow_total" in rp.columns:
            grp = rp.groupby(["week_start", cat_col])["outflow_total"].sum().to_dict()
            rec_outflow_by_week_cat = {
                (_week_key(wk), _cat_key(cat)): float(v)
                for (wk, cat), v in grp.items()
            }

    return patterns, recurring_proj, rec_outflow_by_week, rec_outflow_by_week_cat


def _try_xgb_category_variable_forecast(cat_hist_df, horizon_weeks, recurring_proj):
    """
    Returns: (df_or_none, error_or_none)
    """
    try:
        from app.services.xgb_forecast import forecast_category_variable_xgb
        logger.info("✅ XGB import OK")
    except Exception as e:
        err = f"XGB import failed: {repr(e)}"
        logger.exception(err)
        return None, err

    try:
        df = forecast_category_variable_xgb(
            cat_hist_df=cat_hist_df,
            horizon_weeks=horizon_weeks,
            recurring_proj_df=recurring_proj,
        )
        if df is None or getattr(df, "empty", True):
            return None, "XGB returned empty dataframe"
        logger.info("✅ XGB run OK. rows=%s cols=%s", len(df), list(df.columns))
        return df, None
    except Exception as e:
        err = f"XGB run failed: {repr(e)}"
        logger.exception(err)
        return None, err


def _build_xgb_lookup(xgb_df: pd.DataFrame):
    """
    Build robust lookup:
      (week_key, cat_key) -> {inflow_total, outflow_total, method}
    """
    if xgb_df is None or getattr(xgb_df, "empty", True):
        return {}

    df = xgb_df.copy()
    df["week_start"] = df["week_start"].apply(_week_key)
    df["cat_key"] = df["category_normalized"].apply(_cat_key)

    lookup = {}
    for _, r in df.iterrows():
        wk = r["week_start"]
        ck = r["cat_key"]
        lookup[(wk, ck)] = {
            "inflow_total": float(r.get("inflow_total", 0.0) or 0.0),
            "outflow_total": float(r.get("outflow_total", 0.0) or 0.0),
            "method": str(r.get("method", "baseline") or "baseline").strip().lower(),
        }
    return lookup


def _spec_method_tag(raw_method: str) -> str:
    """
    Spec requires: 'xgboost' or 'baseline' only.
    xgb_forecast may return other strings -> treat as baseline.
    """
    m = (raw_method or "").strip().lower()
    return "xgboost" if m == "xgboost" else "baseline"


# --- Decimal helpers for exact vendor reconciliation ---
def _d2(x) -> Decimal:
    return Decimal(str(float(x)))


def _q2(x: Decimal) -> Decimal:
    return x.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


# -------------------------
# Ingestion
# -------------------------

@router.post("/ingest")
async def ingest(
    file: UploadFile = File(...),
    starting_balance: Optional[float] = Form(None),
):
    """
    starting_balance is provided at ingestion and stored under ingestion_id.
    """
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Upload a .csv file")

    if starting_balance is not None:
        try:
            starting_balance = float(starting_balance)
        except Exception:
            raise HTTPException(status_code=400, detail="starting_balance must be a number")

    raw = await file.read()
    db = get_db()
    ingestion_id = shortuuid.uuid()

    db.ingestions.insert_one({
        "ingestion_id": ingestion_id,
        "filename": file.filename,
        "status": "uploaded",
        "validation_issues": [],
        "record_counts": {"raw_rows": 0, "parsed_rows": 0, "dropped_rows": 0},
        "starting_balance": starting_balance,
    })

    db.raw_files.insert_one({
        "ingestion_id": ingestion_id,
        "filename": file.filename,
        "content_type": file.content_type,
        "raw_csv": Binary(raw),
    })

    try:
        df = load_and_normalize_csv(raw)
    except Exception as e:
        db.ingestions.update_one(
            {"ingestion_id": ingestion_id},
            {"$set": {
                "status": "failed",
                "validation_issues": [str(e)],
                "record_counts": {"raw_rows": 0, "parsed_rows": 0, "dropped_rows": 0},
            }}
        )
        raise HTTPException(status_code=400, detail=f"CSV validation/parsing failed: {e}")

    records = df.to_dict(orient="records")
    for r in records:
        r["ingestion_id"] = ingestion_id

    if records:
        db.transactions.insert_many(records)

    db.ingestions.update_one(
        {"ingestion_id": ingestion_id},
        {"$set": {
            "status": "completed",
            "record_counts": {
                "raw_rows": int(len(df)),
                "parsed_rows": int(len(records)),
                "dropped_rows": int(len(df) - len(records)),
            },
            "starting_balance": starting_balance,
        }}
    )

    return {"ingestion_id": ingestion_id, "starting_balance": starting_balance}


@router.get("/ingest/{ingestion_id}/status")
def ingest_status(ingestion_id: str):
    db = get_db()
    doc = db.ingestions.find_one({"ingestion_id": ingestion_id}, {"_id": 0})
    if not doc:
        raise HTTPException(status_code=404, detail="ingestion_id not found")
    return doc


# -------------------------
# Forecast by category (source of truth)
# -------------------------

@router.get("/forecast/categories")
def forecast_categories(ingestion_id: str):
    db = get_db()
    txns = _get_txns_or_404(db, ingestion_id)
    horizon_weeks = _get_horizon_weeks(txns)

    patterns, recurring_proj, _, rec_outflow_by_week_cat = _build_recurring_maps(txns, horizon_weeks)

    cat_hist = compute_weekly_by_category(txns)
    cat_baseline = baseline_13_week_category_forecast(cat_hist, horizon_weeks).copy()

    cat_baseline["week_start"] = cat_baseline["week_start"].apply(_week_key)
    cat_baseline["cat_key"] = cat_baseline["category_normalized"].apply(_cat_key)

    xgb_variable, xgb_error = _try_xgb_category_variable_forecast(cat_hist, horizon_weeks, recurring_proj)
    xgb_lookup = _build_xgb_lookup(xgb_variable)

    xgb_used_any = False
    if xgb_variable is not None and "method" in getattr(xgb_variable, "columns", []):
        try:
            xgb_used_any = bool((xgb_variable["method"].astype(str).str.lower() == "xgboost").any())
        except Exception:
            xgb_used_any = False

    weeks_payload = []
    for ws in horizon_weeks:
        ws = _week_key(ws)
        base_week = cat_baseline[cat_baseline["week_start"] == ws].copy()

        categories = []
        for _, r in base_week.iterrows():
            cat_norm = str(r["category_normalized"])
            ck = r["cat_key"]

            baseline_in = float(r.get("inflow_total", 0.0) or 0.0)
            baseline_out = float(r.get("outflow_total", 0.0) or 0.0)

            recurring_out = float(rec_outflow_by_week_cat.get((ws, ck), 0.0))

            variable_in = baseline_in
            variable_out = max(0.0, baseline_out - recurring_out)
            method_tag = "baseline"

            if (ws, ck) in xgb_lookup:
                pred = xgb_lookup[(ws, ck)]
                variable_in = float(pred["inflow_total"])
                variable_out = float(pred["outflow_total"])
                method_tag = _spec_method_tag(pred["method"])

            inflow_total = variable_in
            outflow_total = recurring_out + max(0.0, variable_out)

            categories.append({
                "category": cat_norm,
                "inflow_total": round(inflow_total, 2),
                "outflow_total": round(outflow_total, 2),
                "method": method_tag,
                "drivers": {
                    "recurring_outflow": round(recurring_out, 2),
                    "variable_outflow": round(max(0.0, variable_out), 2),
                }
            })

        inflow_sum = round(sum(c["inflow_total"] for c in categories), 2)
        outflow_sum = round(sum(c["outflow_total"] for c in categories), 2)

        weeks_payload.append({
            "week_start": ws.strftime("%Y-%m-%d"),
            "inflow_total": inflow_sum,
            "outflow_total": outflow_sum,
            "categories": categories
        })

    top_method = "hybrid_xgb" if xgb_used_any else "hybrid_baseline"

    db.forecasts.update_one(
        {"ingestion_id": ingestion_id, "type": "categories_hybrid"},
        {"$set": {
            "ingestion_id": ingestion_id,
            "type": "categories_hybrid",
            "method": top_method,
            "use_xgboost": xgb_used_any,
            "xgb_error": xgb_error,
            "weeks": weeks_payload,
            "recurring_patterns": patterns,
        }},
        upsert=True
    )

    db.forecasts.update_one(
        {"ingestion_id": ingestion_id, "type": "recurring_detected"},
        {"$set": {
            "ingestion_id": ingestion_id,
            "type": "recurring_detected",
            "patterns": patterns,
        }},
        upsert=True
    )

    return {
        "ingestion_id": ingestion_id,
        "method": top_method,
        "xgb_used_any": xgb_used_any,
        "xgb_error": xgb_error,
        "weeks": weeks_payload
    }


# -------------------------
# Forecast totals (derived from categories) + QA outputs (3 items)
# -------------------------

@router.get("/forecast")
def forecast(ingestion_id: str):
    db = get_db()

    ing_doc = db.ingestions.find_one({"ingestion_id": ingestion_id}, {"_id": 0})
    if not ing_doc:
        raise HTTPException(status_code=404, detail="ingestion_id not found")

    starting_balance = ing_doc.get("starting_balance", None)
    if starting_balance is None:
        raise HTTPException(
            status_code=400,
            detail="starting_balance is required. Provide it during /cashflow/ingest."
        )
    starting_balance = float(starting_balance)

    cached = db.forecasts.find_one({"ingestion_id": ingestion_id, "type": "totals"}, {"_id": 0})
    if cached and cached.get("weeks"):
        used = cached.get("starting_balance_used", None)
        try:
            used_f = float(used) if used is not None else None
        except Exception:
            used_f = None

        if used_f is not None and abs(used_f - starting_balance) < 1e-9:
            # Ensure QA fields exist on response (older cached docs may not have them)
            qa_checks = cached.get("qa_checks")
            if not qa_checks:
                try:
                    txns = _get_txns_or_404(db, ingestion_id)
                    qa_checks = {}
                    qa_checks.update(historical_min_balance_and_date(txns, starting_balance))
                    qa_checks.update(forecast_min_balance_and_week(cached["weeks"]))
                    qa_checks.update(forecast_outflow_anomaly_warning(txns, cached["weeks"]))
                except Exception as e:
                    logger.warning("QA recompute failed on cached totals: %s", e)
                    qa_checks = {}

            # Keep publish behavior unchanged
            try:
                _ = forecast_vendors(ingestion_id, top_n=10)
                upsert_knowledge_package(ingestion_id)
                publish_to_pinecone(ingestion_id)
            except Exception as e:
                logger.warning("Post-forecast publish skipped/failed (cached totals): %s", e)

            return {
                "ingestion_id": ingestion_id,
                "method": cached.get("method", "baseline"),
                "starting_balance_used": used_f,
                "qa_checks": qa_checks,
                "weeks": cached["weeks"],
            }
        # else mismatch -> rebuild

    txns = _get_txns_or_404(db, ingestion_id)

    cat_resp = forecast_categories(ingestion_id)
    weeks_cat = cat_resp["weeks"]
    method = cat_resp["method"]


    paid_sched = schedule_paid_outflows_with_payables(weeks_cat=weeks_cat, starting_balance=starting_balance)


    weeks_payload = []
    running_balance = starting_balance

    for w in weeks_cat:
        inflow_total = float(w["inflow_total"])
        outflow_total = float(w["outflow_total"])  # intended (reconciles to categories)

        sched = paid_sched[len(weeks_payload)]
        outflow_paid = float(sched["paid_outflow_total"])
        deferred = float(sched["deferred_outflow_total"])
        payables_end = float(sched["payables_queue_balance_end"])

        net_cash = inflow_total - outflow_paid
        running_balance += net_cash

        rec_sum = 0.0
        var_sum = 0.0
        for c in w["categories"]:
            rec_sum += float(c["drivers"]["recurring_outflow"])
            var_sum += float(c["drivers"]["variable_outflow"])

        weeks_payload.append({
            "outflow_total_paid": round(outflow_paid, 2),
            "deferred_outflow_total": round(deferred, 2),
            "payables_queue_balance_end": round(payables_end, 2),
            "week_start": w["week_start"],
            "inflow_total": round(inflow_total, 2),
            "outflow_total": round(outflow_total, 2),
            "net_cash": round(net_cash, 2),
            "ending_balance": round(running_balance, 2),
            "drivers": {
                "recurring_outflow": round(rec_sum, 2),
                "variable_outflow": round(var_sum, 2),
                "note": "Totals reconcile to categories via outflow_total; ending_balance uses outflow_total_paid and defers unpaid variable outflows into a payables queue."
            }
        })

    # ---- REQUIRED 3 QA outputs (only) ----
    qa_checks = {}
    try:
        qa_checks.update(historical_min_balance_and_date(txns, starting_balance))
    except Exception as e:
        logger.warning("QA historical_min_balance_and_date failed: %s", e)

    try:
        qa_checks.update(forecast_min_balance_and_week(weeks_payload))
    except Exception as e:
        logger.warning("QA forecast_min_balance_and_week failed: %s", e)

    try:
        qa_checks.update(forecast_outflow_anomaly_warning(txns, weeks_payload))
    except Exception as e:
        logger.warning("QA forecast_outflow_anomaly_warning failed: %s", e)

    db.forecasts.update_one(
        {"ingestion_id": ingestion_id, "type": "totals"},
        {"$set": {
            "ingestion_id": ingestion_id,
            "type": "totals",
            "method": method,
            "starting_balance_used": starting_balance,
            "qa_checks": qa_checks,
            "weeks": weeks_payload
        }},
        upsert=True
    )

    try:
        _ = forecast_vendors(ingestion_id, top_n=10)
        upsert_knowledge_package(ingestion_id)
        publish_to_pinecone(ingestion_id)
    except Exception as e:
        logger.warning("Post-forecast publish skipped/failed: %s", e)

    return {
        "ingestion_id": ingestion_id,
        "method": method,
        "starting_balance_used": starting_balance,
        "qa_checks": qa_checks,
        "weeks": weeks_payload
    }


# -------------------------
# Forecast by vendor (FORCE EXACT reconciliation to totals outflow)
# -------------------------

@router.get("/forecast/vendors")
def forecast_vendors(ingestion_id: str, top_n: int = 10):
    db = get_db()
    txns = _get_txns_or_404(db, ingestion_id)

    ing_doc = db.ingestions.find_one({"ingestion_id": ingestion_id}, {"_id": 0})
    if not ing_doc:
        raise HTTPException(status_code=404, detail="ingestion_id not found")

    starting_balance = ing_doc.get("starting_balance", None)
    if starting_balance is None:
        raise HTTPException(
            status_code=400,
            detail="starting_balance is required. Provide it during /cashflow/ingest."
        )

    totals_doc = _get_totals_doc(db, ingestion_id, expected_starting_balance=float(starting_balance))
    totals_weeks = totals_doc.get("weeks", [])
    method = totals_doc.get("method", "baseline")

    horizon_weeks = [_week_key(w["week_start"]) for w in totals_weeks]
    totals_outflow_by_week = {
        _week_key(w["week_start"]): _q2(_d2(w["outflow_total"]))
        for w in totals_weeks
    }

    vend_hist = compute_weekly_by_vendor_outflows(txns)
    top_vendors = top_vendors_by_total_outflow(vend_hist, top_n=top_n)

    vend_fcst = baseline_13_week_vendor_forecast(vend_hist, horizon_weeks, top_vendors).copy()
    if not vend_fcst.empty:
        vend_fcst["week_start"] = vend_fcst["week_start"].apply(_week_key)

    response_weeks = []
    for ws in horizon_weeks:
        sub = vend_fcst[vend_fcst["week_start"] == ws].copy()
        total_outflow = totals_outflow_by_week.get(ws, Decimal("0.00"))

        vendors_list = []
        for _, r in sub.iterrows():
            v_amt = _q2(_d2(r["outflow_total"]))
            vendors_list.append({
                "vendor": str(r["vendor_normalized"]),
                "outflow_total": float(v_amt),
                "method": str(r.get("method", "baseline") or "baseline"),
            })

        sum_top = sum(Decimal(str(v["outflow_total"])) for v in vendors_list)

        if sum_top > total_outflow:
            if sum_top > Decimal("0.00"):
                scale = total_outflow / sum_top
                for v in vendors_list:
                    v_new = _q2(Decimal(str(v["outflow_total"])) * scale)
                    v["outflow_total"] = float(v_new)
            other = Decimal("0.00")
        else:
            other = _q2(total_outflow - sum_top)

        current_sum = sum(Decimal(str(v["outflow_total"])) for v in vendors_list) + other
        diff = _q2(total_outflow - current_sum)
        if diff != Decimal("0.00"):
            other = _q2(other + diff)

        vendors_list.append({
            "vendor": "other_vendors",
            "outflow_total": float(other),
            "method": "remainder",
        })

        response_weeks.append({
            "week_start": ws.strftime("%Y-%m-%d"),
            "vendors": vendors_list
        })

    db.forecasts.update_one(
        {"ingestion_id": ingestion_id, "type": "vendors", "top_n": top_n},
        {"$set": {
            "ingestion_id": ingestion_id,
            "type": "vendors",
            "method": method,
            "top_n": top_n,
            "top_vendors": top_vendors,
            "weeks": response_weeks
        }},
        upsert=True
    )

    return {
        "ingestion_id": ingestion_id,
        "method": method,
        "top_n": top_n,
        "top_vendors": top_vendors,
        "weeks": response_weeks
    }


# -------------------------
# Recurring
# -------------------------

@router.get("/recurring")
def recurring(ingestion_id: str):
    db = get_db()
    txns = _get_txns_or_404(db, ingestion_id)

    patterns = detect_recurring_payments(txns)

    db.forecasts.update_one(
        {"ingestion_id": ingestion_id, "type": "recurring_detected"},
        {"$set": {
            "ingestion_id": ingestion_id,
            "type": "recurring_detected",
            "patterns": patterns
        }},
        upsert=True
    )

    return {"ingestion_id": ingestion_id, "patterns": patterns}
