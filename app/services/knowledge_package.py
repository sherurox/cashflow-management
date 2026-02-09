# app/services/knowledge_package.py

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Optional, List
import hashlib
import json

from app.db.mongo import get_db
from app.services.aggregation import (
    compute_weekly_history,
    compute_weekly_by_category,
    compute_weekly_by_vendor_outflows,
)

# Optional QA-check helpers (may or may not exist in your aggregation.py)
# We import them safely so this file won't crash if they aren't present yet.
try:
    from app.services.aggregation import (
        historical_min_balance_and_date,
        forecast_min_balance_and_week,
        forecast_outflow_anomaly_warning,
    )
except Exception:
    historical_min_balance_and_date = None
    forecast_min_balance_and_week = None
    forecast_outflow_anomaly_warning = None


def _stable_hash(obj: Any) -> str:
    """
    Deterministic hash for a JSON-serializable object.
    Used to detect if a package changed between runs.
    """
    s = json.dumps(obj, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(s).hexdigest()


def _extract_qa_checks_from_totals_doc(totals_doc: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Normalize where qa_checks might live.
    We prefer totals_doc["qa_checks"] if present.
    """
    if not totals_doc:
        return {}
    qc = totals_doc.get("qa_checks")
    return qc if isinstance(qc, dict) else {}


def _format_qa_checks_text(qc: Dict[str, Any]) -> str:
    """
    Produce a compact, searchable text block for Pinecone retrieval.
    Keep key names EXACT so questions match.
    """
    if not qc:
        return "QA_CHECKS: none"

    lines = ["QA_CHECKS"]
    # Keep ordering stable/readable
    keys_order = [
        "historical_min_balance",
        "historical_min_balance_date",
        "forecast_min_balance",
        "forecast_min_balance_week",
        "historical_weekly_outflow_max",
        "historical_weekly_outflow_p95",
        "forecast_weekly_outflow_max",
        "forecast_weekly_outflow_max_week",
        "warning_outflow_gt_hist_max",
        "warning_outflow_gt_hist_p95",
    ]
    for k in keys_order:
        if k in qc:
            lines.append(f"{k}: {qc.get(k)}")

    # Include any other extra keys deterministically (if you add more later)
    extras = sorted([k for k in qc.keys() if k not in set(keys_order)])
    for k in extras:
        lines.append(f"{k}: {qc.get(k)}")

    return "\n".join(lines)


def _maybe_compute_qa_checks(
    txns: List[Dict[str, Any]],
    totals_doc: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    If qa_checks is missing from totals_doc, try to compute it
    (only if helper functions exist).
    """
    # If you already stored qa_checks in totals, use it.
    qc = _extract_qa_checks_from_totals_doc(totals_doc)
    if qc:
        return qc

    # No stored qa_checks: attempt recompute if functions exist and totals has weeks
    if not totals_doc or not isinstance(totals_doc.get("weeks"), list):
        return {}

    if (historical_min_balance_and_date is None) or (forecast_min_balance_and_week is None) or (forecast_outflow_anomaly_warning is None):
        # Helpers not available -> cannot compute here
        return {}

    # Need starting balance to compute historical min balance reliably
    starting_balance_used = totals_doc.get("starting_balance_used")
    try:
        starting_balance_used = float(starting_balance_used) if starting_balance_used is not None else None
    except Exception:
        starting_balance_used = None

    if starting_balance_used is None:
        # Without anchor we can't compute meaningful anchored balances
        return {}

    weeks_payload = totals_doc.get("weeks", [])
    qc_out: Dict[str, Any] = {}

    try:
        qc_out.update(historical_min_balance_and_date(txns, starting_balance_used))
    except Exception:
        pass

    try:
        qc_out.update(forecast_min_balance_and_week(weeks_payload))
    except Exception:
        pass

    try:
        qc_out.update(forecast_outflow_anomaly_warning(txns, weeks_payload))
    except Exception:
        pass

    return qc_out


def build_cashflow_knowledge_package(
    ingestion_id: str,
    *,
    ensure_forecasts_exist: bool = False,
) -> Dict[str, Any]:
    """
    Build the structured knowledge package required by the spec.
    Source of truth is Mongo collections already populated by your endpoints.
    """
    db = get_db()

    # ---- raw transactions (traceability) ----
    txns = list(db.transactions.find({"ingestion_id": ingestion_id}, {"_id": 0}))
    if not txns:
        raise ValueError("No transactions found for ingestion_id (cannot build package).")

    # ---- weekly aggregates (traceability + inspectability) ----
    weekly_totals_hist = compute_weekly_history(txns)  # df
    weekly_by_category_hist = compute_weekly_by_category(txns)  # df
    weekly_by_vendor_hist = compute_weekly_by_vendor_outflows(txns)  # df

    # ---- stored forecasts (preferred) ----
    totals_doc = db.forecasts.find_one({"ingestion_id": ingestion_id, "type": "totals"}, {"_id": 0})
    categories_doc = db.forecasts.find_one({"ingestion_id": ingestion_id, "type": "categories_hybrid"}, {"_id": 0})

    # ✅ deterministic vendors artifact: always prefer top_n=10
    vendors_doc = db.forecasts.find_one(
        {"ingestion_id": ingestion_id, "type": "vendors", "top_n": 10},
        {"_id": 0},
    )
    if not vendors_doc:
        vendors_doc = db.forecasts.find_one(
            {"ingestion_id": ingestion_id, "type": "vendors"},
            {"_id": 0},
            sort=[("top_n", 1)],
        )

    recurring_doc = db.forecasts.find_one({"ingestion_id": ingestion_id, "type": "recurring_detected"}, {"_id": 0})

    if ensure_forecasts_exist:
        missing = []
        if not totals_doc:
            missing.append("totals")
        if not categories_doc:
            missing.append("categories_hybrid")
        if not vendors_doc:
            missing.append("vendors (top_n=10)")
        if not recurring_doc:
            missing.append("recurring_detected")
        if missing:
            raise ValueError(f"Missing forecast artifacts in Mongo: {missing}. Run forecast endpoints first.")

    # ---- QA checks (store + make searchable) ----
    qa_checks = _maybe_compute_qa_checks(txns, totals_doc)
    qa_checks_text = _format_qa_checks_text(qa_checks)

    package: Dict[str, Any] = {
        "ingestion_id": ingestion_id,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "artifacts": {
            "totals_forecast": totals_doc,
            "categories_forecast": categories_doc,
            "vendors_forecast": vendors_doc,
            "recurring_patterns": recurring_doc.get("patterns") if recurring_doc else None,
            # ✅ NEW: explicit QA checks artifact (so indexer can include it)
            "qa_checks": qa_checks,
        },
        # ✅ NEW: explicit searchable text chunk (very helpful for Pinecone matching)
        "qa_checks_text": qa_checks_text,
        "traceability": {
            "weekly_history_totals": weekly_totals_hist.to_dict(orient="records"),
            "weekly_history_by_category": weekly_by_category_hist.to_dict(orient="records"),
            "weekly_history_by_vendor_outflows": weekly_by_vendor_hist.to_dict(orient="records"),
            "transactions": txns,
        },
    }

    # Drivers by week
    drivers: List[Dict[str, Any]] = []
    if totals_doc and isinstance(totals_doc.get("weeks"), list):
        for w in totals_doc["weeks"]:
            drivers.append({
                "week_start": w.get("week_start"),
                "drivers": w.get("drivers", {}),
                "inflow_total": w.get("inflow_total"),
                "outflow_total": w.get("outflow_total"),
                # NEW payable-rhythm fields (may be absent in older cached totals)
                "outflow_total_paid": w.get("outflow_total_paid"),
                "deferred_outflow_total": w.get("deferred_outflow_total"),
                "payables_queue_balance_end": w.get("payables_queue_balance_end"),
                "net_cash": w.get("net_cash"),
                "ending_balance": w.get("ending_balance"),
            })
    package["drivers_by_week"] = drivers

    # Stable hash (include qa_checks so changes trigger re-publish)
    package["package_hash"] = _stable_hash({
        "artifacts": package["artifacts"],
        "qa_checks_text": package["qa_checks_text"],
        "drivers_by_week": package["drivers_by_week"],
        "weekly_totals_hist_len": len(package["traceability"]["weekly_history_totals"]),
        "weekly_category_hist_len": len(package["traceability"]["weekly_history_by_category"]),
        "weekly_vendor_hist_len": len(package["traceability"]["weekly_history_by_vendor_outflows"]),
        "txns_len": len(package["traceability"]["transactions"]),
    })

    return package


def upsert_knowledge_package(ingestion_id: str) -> Dict[str, Any]:
    """
    Build package and store it in Mongo collection `knowledge_packages`.
    """
    db = get_db()
    pkg = build_cashflow_knowledge_package(ingestion_id, ensure_forecasts_exist=False)

    db.knowledge_packages.update_one(
        {"ingestion_id": ingestion_id},
        {"$set": pkg},
        upsert=True,
    )
    return pkg


def get_knowledge_package(ingestion_id: str) -> Optional[Dict[str, Any]]:
    db = get_db()
    return db.knowledge_packages.find_one({"ingestion_id": ingestion_id}, {"_id": 0})
