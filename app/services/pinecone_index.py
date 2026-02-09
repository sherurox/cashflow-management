# app/services/pinecone_index.py

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import os
import json

import numpy as np

from app.services.knowledge_package import upsert_knowledge_package


# -----------------------------
# Helpers
# -----------------------------

def _require_env(name: str) -> str:
    v = os.getenv(name)
    if not v:
        raise RuntimeError(f"Missing required env var: {name}")
    return v


def _simple_hash_embedding(text: str, dim: int = 256) -> np.ndarray:
    """
    Fallback embedding when you do not have a real embedding model configured.
    Deterministic and normalized.
    """
    v = np.zeros(dim, dtype=np.float32)
    for i, ch in enumerate(text.encode("utf-8")):
        v[i % dim] += (ch % 13) / 13.0
    n = np.linalg.norm(v) + 1e-9
    return v / n


def _ws_str(x: Any) -> Optional[str]:
    """Force week_start into 'YYYY-MM-DD' string for Pinecone metadata consistency."""
    if x is None:
        return None
    s = str(x).strip()
    return s[:10] if len(s) >= 10 else s


def _deterministic_vector_id(ingestion_id: str, meta: Dict[str, Any], fallback_i: int) -> str:
    """
    Deterministic IDs so we can fetch directly (no metadata filtering dependency).

    - singleton docs always end with ':0'
    - week_totals ends with ':YYYY-MM-DD'
    """
    singleton_types = {
        "summary",
        "qa_checks",
        "categories_forecast",
        "vendors_forecast",
        "recurring_patterns",
    }
    t = str(meta.get("type") or "")
    ws = _ws_str(meta.get("week_start"))

    if t in singleton_types:
        suffix = "0"
    elif ws:
        suffix = ws
    else:
        suffix = str(fallback_i)

    return f"{ingestion_id}:{t}:{suffix}"


def _build_text_chunks(pkg: Dict[str, Any]) -> List[Tuple[str, Dict[str, Any]]]:
    """
    Create a small set of text chunks for retrieval.
    Returns list of (text, metadata) pairs.

    We do NOT embed raw transactions (too big/noisy). We embed summaries + forecasts + drivers + QA checks.
    """
    ingestion_id = pkg["ingestion_id"]
    chunks: List[Tuple[str, Dict[str, Any]]] = []

    artifacts = pkg.get("artifacts", {}) or {}

    # ---- High-level summary chunk ----
    totals = artifacts.get("totals_forecast") or {}
    method = totals.get("method") or (artifacts.get("categories_forecast") or {}).get("method")

    weeks = totals.get("weeks", []) if isinstance(totals.get("weeks"), list) else []
    summary = {
        "ingestion_id": ingestion_id,
        "method": method,
        "num_weeks": len(weeks),
        "first_week": _ws_str(weeks[0].get("week_start")) if weeks else None,
        "last_week": _ws_str(weeks[-1].get("week_start")) if weeks else None,
    }
    chunks.append(
        (
            f"Cashflow forecast summary:\n{json.dumps(summary, indent=2)}",
            {"ingestion_id": ingestion_id, "type": "summary"},
        )
    )

    # ---- QA checks chunk (CRITICAL: index this so Q&A can answer min-balance + anomaly questions) ----
    qa_checks = artifacts.get("qa_checks") or {}
    qa_checks_text = (pkg.get("qa_checks_text") or "").strip()

    if qa_checks_text or qa_checks:
        payload = {
            "qa_checks": qa_checks,
            "qa_checks_text": qa_checks_text if qa_checks_text else None,
        }
        chunks.append(
            (
                "QA checks (quality control metrics):\n" + json.dumps(payload, indent=2, default=str),
                {"ingestion_id": ingestion_id, "type": "qa_checks"},
            )
        )

    # ---- Per-week totals + drivers chunks (SOURCE OF TRUTH: totals_forecast.weeks) ----
    if isinstance(weeks, list) and weeks:
        for w in weeks:
            ws = _ws_str(w.get("week_start"))
            if not ws:
                continue
            chunks.append(
                (
                    "Weekly cashflow totals and drivers:\n"
                    + json.dumps(
                        {
                            "week_start": ws,
                            "drivers": w.get("drivers", {}),
                            "inflow_total": w.get("inflow_total"),
                            "outflow_total": w.get("outflow_total"),
                            # NEW payable-rhythm fields (may be missing for older cached totals)
                            "outflow_total_paid": w.get("outflow_total_paid"),
                            "deferred_outflow_total": w.get("deferred_outflow_total"),
                            "payables_queue_balance_end": w.get("payables_queue_balance_end"),
                            "net_cash": w.get("net_cash"),
                            "ending_balance": w.get("ending_balance"),
                        },
                        indent=2,
                        default=str,
                    ),
                    {"ingestion_id": ingestion_id, "type": "week_totals", "week_start": ws},
                )
            )
    else:
        # Fallback to pkg.drivers_by_week if totals_forecast missing
        for w in pkg.get("drivers_by_week", []) or []:
            ws = _ws_str(w.get("week_start"))
            if not ws:
                continue
            chunks.append(
                (
                    "Weekly cashflow totals and drivers:\n" + json.dumps(w, indent=2, default=str),
                    {"ingestion_id": ingestion_id, "type": "week_totals", "week_start": ws},
                )
            )

    # ---- Categories forecast doc ----
    cat_doc = artifacts.get("categories_forecast")
    if cat_doc:
        chunks.append(
            (
                "Forecast by category (all weeks):\n" + json.dumps(cat_doc, indent=2, default=str),
                {"ingestion_id": ingestion_id, "type": "categories_forecast"},
            )
        )

    # ---- Vendors forecast doc ----
    vend_doc = artifacts.get("vendors_forecast")
    if vend_doc:
        chunks.append(
            (
                "Forecast by vendor (all weeks):\n" + json.dumps(vend_doc, indent=2, default=str),
                {"ingestion_id": ingestion_id, "type": "vendors_forecast"},
            )
        )

    # ---- Recurring patterns doc ----
    rec = artifacts.get("recurring_patterns")
    if rec:
        chunks.append(
            (
                "Detected recurring payment patterns:\n" + json.dumps(rec, indent=2, default=str),
                {"ingestion_id": ingestion_id, "type": "recurring_patterns"},
            )
        )

    return chunks


def _pinecone_client():
    api_key = _require_env("PINECONE_API_KEY")
    from pinecone import Pinecone  # type: ignore
    return Pinecone(api_key=api_key)


def _pinecone_index():
    pc = _pinecone_client()
    index_name = _require_env("PINECONE_INDEX_NAME")
    return pc.Index(index_name)


def _per_ingestion_namespace(ingestion_id: str) -> str:
    base_namespace = os.getenv("PINECONE_NAMESPACE", "cashflow")
    return f"{base_namespace}:{ingestion_id}"


def _fetch_vectors(index, namespace: str, ids: List[str]) -> List[Dict[str, Any]]:
    """
    Fetch by ID (most reliable; no filtering).
    Returns list of {id, score, metadata} for those present.
    """
    if not ids:
        return []
    fr = index.fetch(ids=ids, namespace=namespace)
    vecs = getattr(fr, "vectors", None) or {}

    out: List[Dict[str, Any]] = []
    for _id in ids:
        v = vecs.get(_id)
        if not v:
            continue
        md = getattr(v, "metadata", None) or {}
        out.append({"id": _id, "score": 1.0, "metadata": md})
    return out


# -----------------------------
# Public API
# -----------------------------

def publish_to_pinecone(ingestion_id: str) -> Dict[str, Any]:
    """
    Builds knowledge package, stores in Mongo, then indexes in Pinecone.

    IMPORTANT:
    We use a per-ingestion namespace to avoid relying on metadata filtering
    (which can be limited/unsupported on some Pinecone index tiers).
    """
    pkg = upsert_knowledge_package(ingestion_id)

    # Validate env early (clear error message)
    _require_env("PINECONE_API_KEY")
    index_name = _require_env("PINECONE_INDEX_NAME")

    namespace = _per_ingestion_namespace(ingestion_id)
    index = _pinecone_index()

    chunks = _build_text_chunks(pkg)

    vectors_payload: List[Dict[str, Any]] = []
    for i, (text, meta) in enumerate(chunks):
        vec = _simple_hash_embedding(text, dim=256).tolist()
        vec_id = _deterministic_vector_id(ingestion_id, meta, fallback_i=i)

        meta2 = dict(meta)
        meta2["package_hash"] = pkg.get("package_hash") or ""
        meta2["text"] = text[:2000]  # keep a snippet for inspection in supporting_records

        vectors_payload.append({"id": vec_id, "values": vec, "metadata": meta2})

    # Upsert vectors
    index.upsert(vectors=vectors_payload, namespace=namespace)

    # Verify via fetch of a known deterministic ID (no filters)
    probe_id = f"{ingestion_id}:summary:0"
    found = _fetch_vectors(index, namespace, [probe_id])
    if not found:
        raise RuntimeError(
            "Pinecone upsert verification failed: probe_id not found after upsert. "
            f"probe_id={probe_id} index_name={index_name} namespace={namespace}"
        )

    return {
        "ingestion_id": ingestion_id,
        "index_name": index_name,
        "namespace": namespace,
        "num_vectors_upserted": len(vectors_payload),
        "package_hash": pkg.get("package_hash"),
    }


def retrieve_from_pinecone(
    ingestion_id: str,
    question: str,
    top_k: int = 6,
    week_start: Optional[str] = None,
    allowed_types: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Retrieval strategy (reliable on all Pinecone tiers):
    1) If week_start provided: fetch exact week_totals by deterministic ID.
    2) If allowed_types provided: fetch singleton docs by deterministic IDs first.
    3) Fill remainder with similarity query inside the per-ingestion namespace.
    4) Post-filter in Python (type/week) and dedupe.
    """
    _require_env("PINECONE_API_KEY")
    _require_env("PINECONE_INDEX_NAME")

    namespace = _per_ingestion_namespace(ingestion_id)
    index = _pinecone_index()

    qvec = _simple_hash_embedding(question, dim=256).tolist()

    results: List[Dict[str, Any]] = []
    seen_ids: set[str] = set()

    # 1) Exact week fetch
    if week_start:
        ws = _ws_str(week_start)
        if ws:
            exact_id = f"{ingestion_id}:week_totals:{ws}"
            got = _fetch_vectors(index, namespace, [exact_id])
            for r in got:
                rid = r.get("id") or ""
                if rid and rid not in seen_ids:
                    results.append(r)
                    seen_ids.add(rid)
            if len(results) >= top_k:
                return results[:top_k]

    # 2) Deterministic singleton fetches for allowed_types
    if allowed_types:
        singleton_types = {
            "summary",
            "qa_checks",
            "categories_forecast",
            "vendors_forecast",
            "recurring_patterns",
        }
        ids = []
        for t in allowed_types:
            if t in singleton_types:
                ids.append(f"{ingestion_id}:{t}:0")
        got = _fetch_vectors(index, namespace, ids)
        for r in got:
            rid = r.get("id") or ""
            if rid and rid not in seen_ids:
                results.append(r)
                seen_ids.add(rid)
        if len(results) >= top_k:
            return results[:top_k]

    # 3) Similarity query (no metadata filter; namespace isolates ingestion)
    remaining = top_k - len(results)
    if remaining > 0:
        res = index.query(
            vector=qvec,
            top_k=max(remaining * 4, 12),  # overfetch then filter/dedupe
            include_metadata=True,
            namespace=namespace,
        )
        raw_matches = getattr(res, "matches", None) or []

        for m in raw_matches:
            mid = getattr(m, "id", None)
            if not mid or mid in seen_ids:
                continue
            score = getattr(m, "score", None)
            meta = getattr(m, "metadata", None) or {}
            results.append({"id": mid, "score": score, "metadata": meta})
            seen_ids.add(mid)

    # 4) Post-filter in Python
    if week_start:
        ws = _ws_str(week_start)
        if ws:
            results = [
                x for x in results
                if _ws_str((x.get("metadata") or {}).get("week_start")) == ws
                or (x.get("metadata") or {}).get("type") != "week_totals"
            ]
            # For week_start queries, we really only want week_totals
            results = [x for x in results if (x.get("metadata") or {}).get("type") == "week_totals"]

    if allowed_types:
        allowed = set(allowed_types)
        results = [x for x in results if (x.get("metadata") or {}).get("type") in allowed]

    return results[:top_k]
