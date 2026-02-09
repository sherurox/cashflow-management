# app/routes/qa.py

from __future__ import annotations

import re
from typing import Optional, List, Dict, Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.services.knowledge_package import upsert_knowledge_package
from app.services.pinecone_index import publish_to_pinecone, retrieve_from_pinecone
from app.services.gemini_qa import answer_with_gemini

router = APIRouter()


def extract_week_start(question: str) -> Optional[str]:
    m = re.search(r"\b(20\d{2}-\d{2}-\d{2})\b", question or "")
    return m.group(1) if m else None


def choose_allowed_types(question: str) -> List[str]:
    """
    Deterministic retrieval routing so QA is aligned with the Task Specification:
    - vendor questions: must include vendors + summary (+ optionally week_totals + recurring)
    - category questions: must include categories + summary (+ optionally week_totals + recurring)
    - default: include all main docs
    """
    q = (question or "").lower()

    vendor_keys = ["vendor", "vendors", "supplier", "payee", "merchant"]
    category_keys = [
        "category", "categories", "cogs", "cost of goods", "rent", "utilities",
        "marketing", "labor", "payroll", "insurance", "subscriptions",
    ]
    recurring_keys = ["recurring", "cadence", "monthly", "weekly", "biweekly", "pattern"]

    if any(k in q for k in vendor_keys):
        # enforce summary + vendors_forecast always
        types = ["qa_checks", "summary", "vendors_forecast", "week_totals", "recurring_patterns", "categories_forecast"]
        return types


    if any(k in q for k in category_keys):
        # enforce summary + categories_forecast always
        types = ["qa_checks", "summary", "categories_forecast", "week_totals", "recurring_patterns", "vendors_forecast"]
        return types


    if any(k in q for k in recurring_keys):
        # recurring analysis: ensure recurring + summary present
        types = ["qa_checks", "summary", "recurring_patterns", "week_totals", "categories_forecast", "vendors_forecast"]
        return types


    return ["qa_checks", "summary", "week_totals", "categories_forecast", "vendors_forecast", "recurring_patterns"]



class QARequest(BaseModel):
    ingestion_id: str = Field(..., min_length=5)
    question: str = Field(..., min_length=3)
    top_k: int = Field(6, ge=1, le=20)


@router.post("/qa")
def cashflow_qa(body: QARequest):
    """
    Spec:
      POST /cashflow/qa
      Body: ingestion_id, question
      Returns: answer + supporting_records (+ publishing info)
    """
    ingestion_id = body.ingestion_id
    question = body.question
    top_k = body.top_k
    week_start = extract_week_start(question)

    # 1) Ensure package exists in Mongo for traceability
    try:
        upsert_knowledge_package(ingestion_id)
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Cannot build knowledge package: {e}")

    # 2) Publish to Pinecone (MUST happen here; /forecast may publish too, but QA must be self-contained)
    try:
        pub = publish_to_pinecone(ingestion_id)
    except Exception as e:
        raise HTTPException(
            status_code=501,
            detail=(
                "Pinecone publishing not available. Configure PINECONE_API_KEY and PINECONE_INDEX_NAME. "
                f"Error: {e}"
            ),
        )

    allowed_types = choose_allowed_types(question)

    def _do_retrieve() -> List[Dict[str, Any]]:
        matches: List[Dict[str, Any]] = []

        # A) If a date is present, fetch the exact week_totals doc (deterministic ID fetch path in pinecone_index.py)
        if week_start:
            exact = retrieve_from_pinecone(
                ingestion_id=ingestion_id,
                question=question,
                top_k=1,
                week_start=week_start,
            )
            matches.extend(exact)

        # B) Fill remaining slots with deterministic, question-routed types
        remaining = max(0, top_k - len(matches))
        if remaining > 0:
            extra = retrieve_from_pinecone(
                ingestion_id=ingestion_id,
                question=question,
                top_k=remaining,
                allowed_types=allowed_types,
            )

            # Deduplicate
            seen = {m.get("id") for m in matches if m.get("id")}
            for m in extra:
                mid = m.get("id")
                if mid and mid not in seen:
                    matches.append(m)
                    seen.add(mid)

        # C) Guarantee "summary" is present for *all* questions (Task Spec wants Q&A grounded in stored outputs)
                # C2) Guarantee "qa_checks" is present (QC questions need this as source of truth)
        if not any((m.get("metadata") or {}).get("type") == "qa_checks" for m in matches):
            qc_only = retrieve_from_pinecone(
                ingestion_id=ingestion_id,
                question="qa_checks",
                top_k=1,
                allowed_types=["qa_checks"],
            )
            seen = {m.get("id") for m in matches if m.get("id")}
            for m in qc_only:
                mid = m.get("id")
                if mid and mid not in seen:
                    matches.append(m)
                    seen.add(mid)



        # D) If vendor question, guarantee vendors_forecast is present (even if similarity misses it)
        ql = (question or "").lower()
        if any(k in ql for k in ["vendor", "vendors", "supplier", "payee", "merchant"]):
            if not any((m.get("metadata") or {}).get("type") == "vendors_forecast" for m in matches):
                vend_only = retrieve_from_pinecone(
                    ingestion_id=ingestion_id,
                    question="vendors",
                    top_k=1,
                    allowed_types=["vendors_forecast"],
                )
                seen = {m.get("id") for m in matches if m.get("id")}
                for m in vend_only:
                    mid = m.get("id")
                    if mid and mid not in seen:
                        matches.append(m)

        return matches[:top_k]

    # 3) Retrieve (with one safe retry if Pinecone is temporarily empty)
    try:
        supporting_records = _do_retrieve()

        if not supporting_records:
            _ = publish_to_pinecone(ingestion_id)
            supporting_records = _do_retrieve()

        if not supporting_records:
            raise HTTPException(
                status_code=500,
                detail=(
                    "No supporting records found in Pinecone for this ingestion_id after publish+retry. "
                    "Check that your server process has PINECONE_API_KEY/PINECONE_INDEX_NAME and that "
                    "namespace/type routing matches the indexed metadata."
                ),
            )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pinecone retrieval failed: {e}")

    # 4) Answer with Gemini grounded ONLY in supporting_records
    try:
        answer = answer_with_gemini(question, supporting_records)
    except Exception as e:
        raise HTTPException(
            status_code=501,
            detail=(
                "Gemini not available. Configure GOOGLE_API_KEY (and install google-generativeai). "
                f"Error: {e}"
            ),
        )

    return {
        "ingestion_id": ingestion_id,
        "question": question,
        "answer": answer,
        "supporting_records": supporting_records,
        "publishing": pub,
        "retrieval_policy": {
            "week_start": week_start,
            "allowed_types": allowed_types,
        },
    }
