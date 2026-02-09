# app/services/gemini_qa.py

from __future__ import annotations

from typing import Any, Dict, List
import os
import json


def answer_with_gemini(question: str, supporting_records: List[Dict[str, Any]]) -> str:
    """
    Uses Gemini to answer grounded ONLY in supporting_records.
    Requires GOOGLE_API_KEY.

    Returns answer text.
    """
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("Gemini not configured. Set GOOGLE_API_KEY.")

    # Lazy import
    try:
        import google.generativeai as genai
    except Exception as e:
        raise RuntimeError(f"google-generativeai not installed/importable: {repr(e)}")

    genai.configure(api_key=api_key)

    model_name = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
    model = genai.GenerativeModel(model_name)

    prompt = f"""
You are a cashflow forecasting assistant.

Rules:
- You MUST answer using only the provided supporting records.
- If the records do not contain the answer, say: "I don't have enough information in the retrieved records."
- Be concise, and when possible cite the specific week_start / category / vendor details.

Question:
{question}

Supporting records (JSON):
{json.dumps(supporting_records, indent=2)}
""".strip()

    resp = model.generate_content(prompt)
    return (resp.text or "").strip()
