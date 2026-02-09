import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import hashlib


# -------------------------
# Core helpers (existing)
# -------------------------
def week_start_monday(dt: pd.Timestamp) -> pd.Timestamp:
    # dt is timestamp; normalize to Monday 00:00
    dt = pd.Timestamp(dt).normalize()
    return dt - pd.Timedelta(days=dt.weekday())


def compute_weekly_history(transactions: list[dict]) -> pd.DataFrame:
    """
    Input: transaction docs from Mongo (already normalized)
    Output df with columns:
      week_start, inflow_total, outflow_total, net_cash
    """
    if not transactions:
        return pd.DataFrame(columns=["week_start", "inflow_total", "outflow_total", "net_cash"])

    df = pd.DataFrame(transactions)

    # ensure datetime
    df["date"] = pd.to_datetime(df["date"])
    df["week_start"] = df["date"].apply(week_start_monday)

    # inflow/outflow totals (outflow should be positive magnitude in totals)
    df["inflow_amt"] = df["amount"].where(df["amount"] > 0, 0.0)
    df["outflow_amt"] = (-df["amount"]).where(df["amount"] < 0, 0.0)

    weekly = (
        df.groupby("week_start", as_index=False)[["inflow_amt", "outflow_amt"]]
          .sum()
          .rename(columns={"inflow_amt": "inflow_total", "outflow_amt": "outflow_total"})
          .sort_values("week_start")
    )
    weekly["net_cash"] = weekly["inflow_total"] - weekly["outflow_total"]
    return weekly


def baseline_13_week_forecast_from_history(weekly_hist: pd.DataFrame) -> pd.DataFrame:
    """Creates a simple but non-static baseline forecast.

    Prior behavior repeated the last-8-weeks median for all 13 future weeks, which can look
    "flat" and untrustworthy.

    New behavior bootstraps (samples with replacement) from the last up to 8 historical weeks,
    preserving the joint (inflow,outflow) structure for realism while remaining deterministic.
    """
    if weekly_hist.empty:
        # return 13 weeks of zeros starting next Monday
        start = week_start_monday(pd.Timestamp(datetime.utcnow())) + pd.Timedelta(days=7)
        weeks = [start + pd.Timedelta(days=7 * i) for i in range(13)]
        out = pd.DataFrame({"week_start": weeks, "inflow_total": 0.0, "outflow_total": 0.0})
        out["net_cash"] = 0.0
        return out

    hist = weekly_hist.copy().sort_values("week_start")
    tail = hist.tail(8).reset_index(drop=True)

    last_week = pd.Timestamp(hist["week_start"].max())
    start = last_week + pd.Timedelta(days=7)
    weeks = [start + pd.Timedelta(days=7 * i) for i in range(13)]

    # deterministic seed based on horizon start date
    seed = int(pd.Timestamp(start).value % (2**32))
    rng = np.random.default_rng(seed)

    idx = rng.integers(0, len(tail), size=len(weeks))
    sampled = tail.loc[idx, ["inflow_total", "outflow_total"]].reset_index(drop=True)

    out = pd.DataFrame({"week_start": weeks})
    out["inflow_total"] = sampled["inflow_total"].astype(float).values
    out["outflow_total"] = sampled["outflow_total"].astype(float).values
    out["net_cash"] = out["inflow_total"] - out["outflow_total"]
    return out


def ending_balance_index(forecast_df: pd.DataFrame, base: float = 0.0) -> pd.Series:
    # Balance index (spec allowed when no starting balance)
    return base + forecast_df["net_cash"].cumsum()


def compute_weekly_by_category(transactions: list[dict]) -> pd.DataFrame:
    """
    Returns a dataframe with:
      week_start, category_normalized, inflow_total, outflow_total
    """
    if not transactions:
        return pd.DataFrame(columns=["week_start", "category_normalized", "inflow_total", "outflow_total"])

    df = pd.DataFrame(transactions)
    df["date"] = pd.to_datetime(df["date"])
    df["week_start"] = df["date"].apply(week_start_monday)

    # choose normalized category; fallback if empty
    df["category_normalized"] = df.get("category_normalized", "").astype(str)
    df.loc[df["category_normalized"].str.strip() == "", "category_normalized"] = "uncategorized"

    df["inflow_amt"] = df["amount"].where(df["amount"] > 0, 0.0)
    df["outflow_amt"] = (-df["amount"]).where(df["amount"] < 0, 0.0)

    cat_week = (
        df.groupby(["week_start", "category_normalized"], as_index=False)[["inflow_amt", "outflow_amt"]]
          .sum()
          .rename(columns={"inflow_amt": "inflow_total", "outflow_amt": "outflow_total"})
          .sort_values(["week_start", "category_normalized"])
    )
    return cat_week


def baseline_13_week_category_forecast(cat_week_hist: pd.DataFrame, horizon_weeks: list[pd.Timestamp]) -> pd.DataFrame:
    """Category baseline forecast with variance.

    Prior behavior used a constant median per category for all future weeks.

    New behavior bootstraps from each category's last up to 8 observed weekly rows
    (sampling week rows with replacement) to introduce realistic variance while preserving
    the joint (inflow,outflow) relationship.

    Returns rows: week_start, category_normalized, inflow_total, outflow_total, method
    """
    if cat_week_hist.empty:
        return pd.DataFrame(columns=["week_start", "category_normalized", "inflow_total", "outflow_total", "method"])

    hist = cat_week_hist.copy()
    hist["week_start"] = pd.to_datetime(hist["week_start"])

    # deterministic seed based on first horizon week (stable per ingestion run)
    seed = int(pd.Timestamp(horizon_weeks[0]).value % (2**32)) if horizon_weeks else 42
    rng = np.random.default_rng(seed)

    categories = sorted(hist["category_normalized"].unique().tolist())

    rows = []
    for cat in categories:
        sub = hist[hist["category_normalized"] == cat].sort_values("week_start")
        tail = sub.tail(8).reset_index(drop=True)
        if tail.empty:
            for ws in horizon_weeks:
                rows.append({
                    "week_start": ws,
                    "category_normalized": cat,
                    "inflow_total": 0.0,
                    "outflow_total": 0.0,
                    "method": "baseline",
                })
            continue

        idx = rng.integers(0, len(tail), size=len(horizon_weeks))
        sampled = tail.loc[idx, ["inflow_total", "outflow_total"]].reset_index(drop=True)

        for i, ws in enumerate(horizon_weeks):
            rows.append({
                "week_start": ws,
                "category_normalized": cat,
                "inflow_total": float(sampled.loc[i, "inflow_total"]),
                "outflow_total": float(sampled.loc[i, "outflow_total"]),
                "method": "baseline",
            })

    return pd.DataFrame(rows)


def compute_weekly_by_vendor_outflows(transactions: list[dict]) -> pd.DataFrame:
    """
    Returns df with:
      week_start, vendor_normalized, outflow_total
    Only outflows (spend) are included.
    """
    if not transactions:
        return pd.DataFrame(columns=["week_start", "vendor_normalized", "outflow_total"])

    df = pd.DataFrame(transactions)
    df["date"] = pd.to_datetime(df["date"])
    df["week_start"] = df["date"].apply(week_start_monday)

    df["vendor_normalized"] = df.get("vendor_normalized", "").astype(str)
    df.loc[df["vendor_normalized"].str.strip() == "", "vendor_normalized"] = "unknown"

    # outflow magnitude
    df["outflow_amt"] = (-df["amount"]).where(df["amount"] < 0, 0.0)
    df = df[df["outflow_amt"] > 0]

    vend_week = (
        df.groupby(["week_start", "vendor_normalized"], as_index=False)[["outflow_amt"]]
          .sum()
          .rename(columns={"outflow_amt": "outflow_total"})
          .sort_values(["week_start", "vendor_normalized"])
    )
    return vend_week


def top_vendors_by_total_outflow(vend_week_hist: pd.DataFrame, top_n: int = 10) -> list[str]:
    if vend_week_hist.empty:
        return []
    totals = (
        vend_week_hist.groupby("vendor_normalized", as_index=False)["outflow_total"]
        .sum()
        .sort_values("outflow_total", ascending=False)
    )
    return totals["vendor_normalized"].head(top_n).tolist()


def baseline_13_week_vendor_forecast(
    vend_week_hist: pd.DataFrame,
    horizon_weeks: list[pd.Timestamp],
    top_vendors: list[str],
) -> pd.DataFrame:
    """Vendor baseline forecast with variance.

    Prior behavior used a constant median spend per vendor for all future weeks.

    New behavior bootstraps from each vendor's last up to 8 observed weekly outflow rows
    (sampling with replacement) to introduce realistic variance.

    Rows: week_start, vendor_normalized, outflow_total, method
    """
    rows = []
    if vend_week_hist.empty or not top_vendors:
        return pd.DataFrame(columns=["week_start", "vendor_normalized", "outflow_total", "method"])

    hist = vend_week_hist.copy()
    hist["week_start"] = pd.to_datetime(hist["week_start"])

    # deterministic seed based on first horizon week
    seed = int(pd.Timestamp(horizon_weeks[0]).value % (2**32)) if horizon_weeks else 42
    rng = np.random.default_rng(seed)

    for v in top_vendors:
        sub = hist[hist["vendor_normalized"] == v].sort_values("week_start")
        tail = sub.tail(8).reset_index(drop=True)
        if tail.empty:
            for ws in horizon_weeks:
                rows.append({
                    "week_start": ws,
                    "vendor_normalized": v,
                    "outflow_total": 0.0,
                    "method": "baseline",
                })
            continue

        idx = rng.integers(0, len(tail), size=len(horizon_weeks))
        sampled = tail.loc[idx, ["outflow_total"]].reset_index(drop=True)

        for i, ws in enumerate(horizon_weeks):
            rows.append({
                "week_start": ws,
                "vendor_normalized": v,
                "outflow_total": float(sampled.loc[i, "outflow_total"]),
                "method": "baseline",
            })

    return pd.DataFrame(rows)

def schedule_paid_outflows_with_payables(
    weeks_cat: list[dict],
    starting_balance: float,
    max_deferral_weeks: int = 4,
) -> list[dict]:
    """
    Payable rhythm / cash constraint:
      - recurring_outflow is fixed (must-pay)
      - variable_outflow can be deferred if cash is tight
      - never allow paid outflows to exceed available cash (balance + inflow)
      - deferred outflows accumulate into a payables queue
    """
    payables_queue: list[dict] = []  # [{"amount": float, "age": int}]
    balance = float(starting_balance)

    out: list[dict] = []

    for w in (weeks_cat or []):
        inflow = float(w.get("inflow_total", 0.0) or 0.0)

        rec_sum = 0.0
        var_sum = 0.0
        for c in (w.get("categories", []) or []):
            d = c.get("drivers", {}) or {}
            rec_sum += float(d.get("recurring_outflow", 0.0) or 0.0)
            var_sum += float(d.get("variable_outflow", 0.0) or 0.0)

        available_cash = balance + inflow

        # 1) Pay recurring first
        paid_rec = min(rec_sum, max(0.0, available_cash))
        available_cash -= paid_rec

        # 2) Pay older deferred payables next (oldest first)
        paid_from_queue = 0.0
        new_queue: list[dict] = []
        for item in payables_queue:
            amt = float(item.get("amount", 0.0) or 0.0)
            age = int(item.get("age", 0) or 0) + 1

            if available_cash <= 0:
                new_queue.append({"amount": amt, "age": age})
                continue

            pay = min(amt, available_cash)
            paid_from_queue += pay
            available_cash -= pay
            rem = amt - pay
            if rem > 1e-9:
                new_queue.append({"amount": rem, "age": age})

        payables_queue = new_queue

        # 3) Pay this week's variable if possible; defer remainder
        paid_var = min(var_sum, max(0.0, available_cash))
        available_cash -= paid_var

        deferred_this_week = max(0.0, var_sum - paid_var)
        if deferred_this_week > 1e-9:
            payables_queue.append({"amount": deferred_this_week, "age": 0})

        # cap ages for reporting only
        for item in payables_queue:
            item["age"] = min(int(item.get("age", 0) or 0), int(max_deferral_weeks))

        paid_outflow_total = paid_rec + paid_from_queue + paid_var
        balance = balance + inflow - paid_outflow_total

        out.append({
            "paid_outflow_total": round(float(paid_outflow_total), 2),
            "deferred_outflow_total": round(float(deferred_this_week), 2),
            "payables_queue_balance_end": round(float(sum(i.get("amount", 0.0) or 0.0 for i in payables_queue)), 2),
        })

    return out

def detect_recurring_payments(transactions: list[dict]) -> list[dict]:
    """
    Simple recurring detector:
    - only outflows
    - group by vendor_normalized
    - cadence based on median day gap (weekly/biweekly/monthly)
    - require >= 3 occurrences
    """
    if not transactions:
        return []

    df = pd.DataFrame(transactions)
    df["date"] = pd.to_datetime(df["date"])
    df["vendor_normalized"] = df.get("vendor_normalized", "").astype(str)
    df.loc[df["vendor_normalized"].str.strip() == "", "vendor_normalized"] = "unknown"

    df["category_normalized"] = df.get("category_normalized", "").astype(str)
    df.loc[df["category_normalized"].str.strip() == "", "category_normalized"] = "uncategorized"

    # outflows only (spend)
    df["outflow_amt"] = (-df["amount"]).where(df["amount"] < 0, 0.0)
    df = df[df["outflow_amt"] > 0].copy()

    if df.empty:
        return []

    results = []

    for vendor, g in df.groupby("vendor_normalized"):
        g = g.sort_values("date")
        if len(g) < 3:
            continue

        # gaps in days
        gaps = g["date"].diff().dt.days.dropna()
        if len(gaps) < 2:
            continue

        med_gap = float(gaps.median())

        cadence = None
        if 6 <= med_gap <= 8:
            cadence = "weekly"
        elif 12 <= med_gap <= 16:
            cadence = "biweekly"
        elif 26 <= med_gap <= 35:
            cadence = "monthly"
        else:
            continue  # not clearly recurring

        # amount stability
        amounts = g["outflow_amt"].astype(float)
        p25 = float(amounts.quantile(0.25))
        p75 = float(amounts.quantile(0.75))
        median_amt = float(amounts.median())

        # choose "most common" category for the vendor
        top_cat = (
            g["category_normalized"].value_counts().idxmax()
            if not g["category_normalized"].isna().all()
            else "uncategorized"
        )

        last_seen = pd.Timestamp(g["date"].max())
        next_expected = last_seen + pd.Timedelta(days=round(med_gap))

        results.append({
            "vendor": vendor,
            "category": top_cat,
            "cadence": cadence,
            "median_gap_days": round(med_gap, 2),
            "typical_amount_range": [round(p25, 2), round(p75, 2)],
            "typical_amount_median": round(median_amt, 2),
            "count": int(len(g)),
            "last_seen_date": last_seen.strftime("%Y-%m-%d"),
            "next_expected_date": next_expected.strftime("%Y-%m-%d"),
        })

    # sort biggest recurring by median spend
    results.sort(key=lambda x: x["typical_amount_median"], reverse=True)
    return results


def project_recurring_to_weeks(patterns: list[dict], horizon_weeks: list[pd.Timestamp]) -> pd.DataFrame:
    """
    Projects recurring patterns onto forecast weeks.
    Returns rows: week_start, vendor, category, outflow_total, method="recurring"
    """
    if not patterns:
        return pd.DataFrame(columns=["week_start", "vendor", "category", "outflow_total", "method"])

    horizon_set = set(pd.Timestamp(w).normalize() for w in horizon_weeks)
    rows = []

    for p in patterns:
        cadence = p.get("cadence")
        vendor = p.get("vendor")
        category = p.get("category", "uncategorized")
        median_amt = float(p.get("typical_amount_median", 0.0))
        amt_range = p.get("typical_amount_range") or []
        p25 = float(amt_range[0]) if isinstance(amt_range, (list, tuple)) and len(amt_range) >= 2 else None
        p75 = float(amt_range[1]) if isinstance(amt_range, (list, tuple)) and len(amt_range) >= 2 else None

        # stable per-vendor seed (avoid Python's randomized hash)
        seed = int(hashlib.md5(str(vendor).encode("utf-8")).hexdigest()[:8], 16)
        rng = np.random.default_rng(seed)

        def sample_amt() -> float:
            if p25 is not None and p75 is not None and p75 >= p25 and p75 > 0:
                return float(rng.uniform(p25, p75))
            return float(median_amt)

        step_days = 7 if cadence == "weekly" else 14 if cadence == "biweekly" else None

        # start from next_expected_date and keep adding cadence
        next_dt = pd.Timestamp(p.get("next_expected_date"))
        if pd.isna(next_dt) or median_amt <= 0:
            continue

        if not step_days and cadence != "monthly":
            continue

        # project forward until we pass horizon end
        max_week = max(horizon_weeks)
        cur = next_dt

        while cur <= (max_week + pd.Timedelta(days=7)):
            ws = week_start_monday(cur)
            ws_norm = pd.Timestamp(ws).normalize()
            if ws_norm in horizon_set:
                rows.append({
                    "week_start": ws_norm,
                    "vendor": vendor,
                    "category": category,
                    "outflow_total": round(sample_amt(), 2),
                    "method": "recurring"
                })
            if cadence == "monthly":
                cur = cur + pd.DateOffset(months=1)
            else:
                cur = cur + pd.Timedelta(days=step_days)

    return pd.DataFrame(rows)


def subtract_recurring_from_category_baseline(
    cat_baseline_fcst: pd.DataFrame,
    recurring_proj: pd.DataFrame
) -> pd.DataFrame:
    """
    Variable component baseline = category baseline - recurring amounts allocated to that category.
    Floors at 0 to prevent negative variable forecasts.
    cat_baseline_fcst rows: week_start, category_normalized, inflow_total, outflow_total, method
    recurring_proj rows: week_start, category, outflow_total
    """
    if cat_baseline_fcst.empty:
        return cat_baseline_fcst

    variable = cat_baseline_fcst.copy()
    variable["method"] = "baseline_variable"

    if recurring_proj is None or recurring_proj.empty:
        return variable

    rec_cat = (
        recurring_proj.groupby(["week_start", "category"], as_index=False)["outflow_total"]
        .sum()
        .rename(columns={"category": "category_normalized", "outflow_total": "recurring_outflow"})
    )

    merged = variable.merge(rec_cat, on=["week_start", "category_normalized"], how="left")
    merged["recurring_outflow"] = merged["recurring_outflow"].fillna(0.0)

    merged["outflow_total"] = (merged["outflow_total"] - merged["recurring_outflow"]).clip(lower=0.0)
    merged = merged.drop(columns=["recurring_outflow"])
    return merged


def recurring_outflow_by_category_week(recurring_proj: pd.DataFrame) -> pd.DataFrame:
    """
    recurring_proj rows: week_start, category, outflow_total
    returns: week_start, category_normalized, recurring_outflow
    """
    if recurring_proj is None or recurring_proj.empty:
        return pd.DataFrame(columns=["week_start", "category_normalized", "recurring_outflow"])

    return (
        recurring_proj.groupby(["week_start", "category"], as_index=False)["outflow_total"]
        .sum()
        .rename(columns={"category": "category_normalized", "outflow_total": "recurring_outflow"})
    )


# -------------------------
# NEW: CFO-style QA helpers (only what you requested)
# -------------------------
def compute_historical_daily_balance_series(
    transactions: list[dict],
    ending_balance_anchor: float,
) -> pd.DataFrame:
    """
    Builds a DAILY anchored balance series for the historical CSV period.

    Interpretation (matches your current system):
      - ending_balance_anchor is the bank balance AFTER the historical period ends
        (i.e., the balance you want to start forecasting from).

    Steps:
      1) Group transactions by calendar date and sum signed 'amount' => daily_net_cash
      2) Compute implied start balance:
           start_balance = ending_balance_anchor - sum(daily_net_cash)
      3) Compute daily ending balance:
           ending_balance = start_balance + cumsum(daily_net_cash)

    Output columns:
      date, daily_net_cash, ending_balance
    """
    if not transactions:
        return pd.DataFrame(columns=["date", "daily_net_cash", "ending_balance"])

    df = pd.DataFrame(transactions).copy()
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()

    # signed sum per day
    daily = (
        df.groupby("date", as_index=False)["amount"]
          .sum()
          .rename(columns={"amount": "daily_net_cash"})
          .sort_values("date")
    )

    total_hist_net = float(daily["daily_net_cash"].sum())
    end_bal = float(ending_balance_anchor)
    start_bal = end_bal - total_hist_net

    daily["ending_balance"] = start_bal + daily["daily_net_cash"].cumsum()
    return daily


def historical_min_balance_and_date(
    transactions: list[dict],
    ending_balance_anchor: float,
) -> dict:
    """
    Returns:
      historical_min_balance: float
      historical_min_balance_date: 'YYYY-MM-DD' (calendar date)
    """
    daily = compute_historical_daily_balance_series(transactions, ending_balance_anchor)
    if daily.empty:
        return {"historical_min_balance": 0.0, "historical_min_balance_date": None}

    idx = daily["ending_balance"].astype(float).idxmin()
    min_row = daily.loc[idx]
    return {
        "historical_min_balance": round(float(min_row["ending_balance"]), 2),
        "historical_min_balance_date": pd.Timestamp(min_row["date"]).strftime("%Y-%m-%d"),
    }


def forecast_min_balance_and_week(forecast_weeks: list[dict]) -> dict:
    """
    forecast_weeks: list of totals weeks dicts (each has 'week_start' and 'ending_balance')

    Returns:
      forecast_min_balance: float
      forecast_min_balance_week: 'YYYY-MM-DD' (week_start)
    """
    if not forecast_weeks:
        return {"forecast_min_balance": 0.0, "forecast_min_balance_week": None}

    df = pd.DataFrame(forecast_weeks).copy()
    if df.empty or "ending_balance" not in df.columns or "week_start" not in df.columns:
        return {"forecast_min_balance": 0.0, "forecast_min_balance_week": None}

    df["week_start"] = pd.to_datetime(df["week_start"])
    df["ending_balance"] = pd.to_numeric(df["ending_balance"], errors="coerce")
    df = df.dropna(subset=["ending_balance", "week_start"])
    if df.empty:
        return {"forecast_min_balance": 0.0, "forecast_min_balance_week": None}

    idx = df["ending_balance"].idxmin()
    r = df.loc[idx]
    return {
        "forecast_min_balance": round(float(r["ending_balance"]), 2),
        "forecast_min_balance_week": pd.Timestamp(r["week_start"]).strftime("%Y-%m-%d"),
    }


def historical_weekly_outflow_stats(transactions: list[dict]) -> dict:
    """
    Computes historical weekly outflow distribution from raw transactions.

    Returns:
      historical_weekly_outflow_max: float
      historical_weekly_outflow_p95: float
    """
    weekly = compute_weekly_history(transactions)
    if weekly.empty or "outflow_total" not in weekly.columns:
        return {"historical_weekly_outflow_max": 0.0, "historical_weekly_outflow_p95": 0.0}

    s = pd.to_numeric(weekly["outflow_total"], errors="coerce").dropna()
    if s.empty:
        return {"historical_weekly_outflow_max": 0.0, "historical_weekly_outflow_p95": 0.0}

    max_v = float(s.max())
    # If only a few weeks exist, quantile is still defined; this is OK for a basic QA check.
    p95_v = float(s.quantile(0.95))

    return {
        "historical_weekly_outflow_max": round(max_v, 2),
        "historical_weekly_outflow_p95": round(p95_v, 2),
    }


def forecast_outflow_anomaly_warning(
    transactions: list[dict],
    forecast_weeks: list[dict],
) -> dict:
    """
    Warning if ANY forecast weekly outflow exceeds historical max or historical p95.

    Returns:
      warning_outflow_gt_hist_max: bool
      warning_outflow_gt_hist_p95: bool
      forecast_weekly_outflow_max: float
      forecast_weekly_outflow_max_week: 'YYYY-MM-DD'
      historical_weekly_outflow_max: float
      historical_weekly_outflow_p95: float
    """
    stats = historical_weekly_outflow_stats(transactions)
    hist_max = float(stats["historical_weekly_outflow_max"])
    hist_p95 = float(stats["historical_weekly_outflow_p95"])

    if not forecast_weeks:
        return {
            **stats,
            "warning_outflow_gt_hist_max": False,
            "warning_outflow_gt_hist_p95": False,
            "forecast_weekly_outflow_max": 0.0,
            "forecast_weekly_outflow_max_week": None,
        }

    df = pd.DataFrame(forecast_weeks).copy()
    if df.empty or "week_start" not in df.columns:
        return {
            **stats,
            "warning_outflow_gt_hist_max": False,
            "warning_outflow_gt_hist_p95": False,
            "forecast_weekly_outflow_max": 0.0,
            "forecast_weekly_outflow_max_week": None,
        }

    df["week_start"] = pd.to_datetime(df["week_start"])

    out_col = "outflow_total_paid" if "outflow_total_paid" in df.columns else "outflow_total"
    if out_col not in df.columns:
        return {
            **stats,
            "warning_outflow_gt_hist_max": False,
            "warning_outflow_gt_hist_p95": False,
            "forecast_weekly_outflow_max": 0.0,
            "forecast_weekly_outflow_max_week": None,
        }
    df[out_col] = pd.to_numeric(df[out_col], errors="coerce")

    df = df.dropna(subset=["week_start", out_col])
    if df.empty:
        return {
            **stats,
            "warning_outflow_gt_hist_max": False,
            "warning_outflow_gt_hist_p95": False,
            "forecast_weekly_outflow_max": 0.0,
            "forecast_weekly_outflow_max_week": None,
        }

    idx = df[out_col].idxmax()
    max_row = df.loc[idx]
    fc_max = float(max_row[out_col])
    fc_week = pd.Timestamp(max_row["week_start"]).strftime("%Y-%m-%d")

    return {
        **stats,
        "forecast_weekly_outflow_max": round(fc_max, 2),
        "forecast_weekly_outflow_max_week": fc_week,
        "warning_outflow_gt_hist_max": bool(hist_max > 0 and fc_max > hist_max),
        "warning_outflow_gt_hist_p95": bool(hist_p95 > 0 and fc_max > hist_p95),
    }
