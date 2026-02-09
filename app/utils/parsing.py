from __future__ import annotations

from io import BytesIO
from datetime import datetime, timezone
from typing import Optional
import pandas as pd
import re
from decimal import Decimal, InvalidOperation


def parse_amount(x) -> float:
    """Parse a money/amount field into a float.

    Handles common bank export formats:
      - $5,209.32   -> 5209.32
      - ($5,903.09) -> -5903.09
      - -5,903.09   -> -5903.09
      - 5 903.09    -> 5903.09

    Notes:
      - Returns 0.0 for blanks/NaN/unparseable values.
      - Uses Decimal for safer parsing then converts to float.
    """
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return 0.0

    s = str(x).strip()
    if s == "" or s.lower() == "nan":
        return 0.0

    # parenthesis means negative in many accounting exports
    neg = s.startswith("(") and s.endswith(")")
    if neg:
        s = s[1:-1].strip()

    # remove currency symbols and separators
    s = s.replace("$", "").replace(",", "").replace(" ", "").strip()

    if s == "":
        return 0.0

    try:
        val = Decimal(s)
    except (InvalidOperation, ValueError):
        return 0.0

    out = float(val)
    return -out if neg else out


def direction_from_amount(amount: float) -> str:
    # if amount == 0, treat as outflow-neutral; keep "outflow" for simplicity
    return "inflow" if amount > 0 else "outflow"


def _normalize_tx_type(s: str) -> str:
    return str(s or "").strip().lower()


def direction_from_transaction_type(tx_type: str) -> Optional[str]:
    """
    Map bank export 'Transaction Type' to inflow/outflow.
    If unknown or empty, return None and fall back to amount sign.
    """
    t = _normalize_tx_type(tx_type)
    if not t:
        return None

    inflow_keys = [
        "deposit",
        "credit",
        "ach credit",
        "transfer in",
        "refund",
        "interest",
    ]

    outflow_keys = [
        "electronic withdrawal",
        "withdrawal",
        "debit",
        "check",
        "cc payment",
        "card payment",
        "payment",
        "fee",
        "charge",
        "ach debit",
        "transfer out",
    ]

    if any(k in t for k in inflow_keys):
        return "inflow"
    if any(k in t for k in outflow_keys):
        return "outflow"
    return None


def normalize_vendor(description: str) -> str:
    """
    Simple deterministic vendor normalization from Description.
    Good enough for v1; refine later.
    """
    s = str(description).strip().lower()

    # remove any obvious trailing metadata patterns
    s = s.split(" orig co name:")[0]
    s = s.split(" memo:")[0]

    # remove leading numeric IDs
    s = re.sub(r"^\d+\s*", "", s)

    # keep alphanum + spaces
    s = re.sub(r"[^a-z0-9\s&\-]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()

    toks = s.split()
    return " ".join(toks[:4]) if toks else "unknown"


def _canonicalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Create a canonical schema for varying bank CSV exports.

    We accept common variations in header casing and punctuation and map them to the
    canonical names used by the rest of the pipeline:
      Date, Amount, Description, Transaction Type, Category, Sub-Category
    """
    def norm(c: str) -> str:
        return re.sub(r"[^a-z0-9]+", "", str(c).strip().lower())

    col_map = {}
    for c in df.columns:
        k = norm(c)
        if k in ("date", "transactiondate", "postingdate"):
            col_map[c] = "Date"
        elif k in ("amount", "amt", "transactionamount"):
            col_map[c] = "Amount"
        elif k in ("description", "desc", "memo", "details", "narrative"):
            col_map[c] = "Description"
        elif k in ("transactiontype", "type"):
            col_map[c] = "Transaction Type"
        elif k in ("category", "cat"):
            col_map[c] = "Category"
        elif k in ("subcategory", "subcat", "subcategoryname", "subcatname", "subcategory"):
            col_map[c] = "Sub-Category"

    df = df.rename(columns=col_map)
    # Strip all column names after rename for safety
    df.columns = [str(c).strip() for c in df.columns]
    return df


def load_and_normalize_csv(file_bytes: bytes) -> pd.DataFrame:
    df = pd.read_csv(BytesIO(file_bytes))

    df = _canonicalize_columns(df)

    # Validate required
    required_cols = ["Date", "Amount", "Description"]
    for c in required_cols:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}. Found: {list(df.columns)}")

    # Category requirement: category OR sub_category must exist
    if ("Category" not in df.columns) and ("Sub-Category" not in df.columns):
        raise ValueError("Missing Category/Sub-Category columns")

    # Parse dates robustly, but prefer common bank export formats to keep parsing consistent
    # and avoid pandas' per-element fallback warning.
    s_dates = df["Date"]
    dt = pd.to_datetime(s_dates, format="%m/%d/%y", errors="coerce")
    if dt.isna().any():
        dt2 = pd.to_datetime(s_dates, format="%m/%d/%Y", errors="coerce")
        # If the 4-digit year format is strictly better, use it.
        if dt2.notna().sum() > dt.notna().sum():
            dt = dt2
    # Final fallback for uncommon formats
    if dt.isna().any():
        dt = pd.to_datetime(s_dates, errors="coerce")
    if dt.isna().any():
        bad = df.loc[dt.isna(), ["Date"]].head(5).to_dict(orient="records")
        raise ValueError(f"Some Date values could not be parsed. Examples: {bad}")

    # Convert to Mongo-safe timezone-aware datetime at midnight UTC
    # Convert to Mongo-safe timezone-aware datetime at midnight UTC
    dt_utc = dt.dt.tz_localize(timezone.utc, nonexistent="shift_forward", ambiguous="NaT")
    df["date"] = dt_utc.dt.normalize()

    # Parse amount (raw; may not always match direction in some bank exports)
    df["amount"] = df["Amount"].apply(parse_amount).astype(float)

    # Capture transaction type if present
    if "Transaction Type" in df.columns:
        df["transaction_type"] = df["Transaction Type"].astype(str)
        tx_dir = df["transaction_type"].apply(direction_from_transaction_type)
        amt_dir = df["amount"].apply(direction_from_amount)
        df["direction"] = tx_dir.fillna(amt_dir)
    else:
        df["transaction_type"] = ""
        df["direction"] = df["amount"].apply(direction_from_amount)

    # Normalize amount to match direction:
    # inflow -> positive, outflow -> negative
    df.loc[(df["direction"] == "outflow") & (df["amount"] > 0), "amount"] *= -1
    df.loc[(df["direction"] == "inflow") & (df["amount"] < 0), "amount"] *= -1

    # Inflow/outflow split (aggregation-ready)
    df["inflow"] = df["amount"].clip(lower=0)
    df["outflow"] = (-df["amount"]).clip(lower=0)

    # Core fields
    df["description"] = df["Description"].astype(str)

    df["category"] = df["Category"].astype(str) if "Category" in df.columns else ""
    df["sub_category"] = df["Sub-Category"].astype(str) if "Sub-Category" in df.columns else ""

    # Normalized fields (spec)
    df["vendor_normalized"] = df["description"].apply(normalize_vendor)
    df["category_normalized"] = df["category"].astype(str).str.strip().str.lower()
    df["sub_category_normalized"] = df["sub_category"].astype(str).str.strip().str.lower()

    # Compatibility fields (existing aggregation code likely expects these)
    df["vendor"] = df["vendor_normalized"]

    return df
