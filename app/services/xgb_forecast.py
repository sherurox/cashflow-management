# app/services/xgb_forecast.py

from __future__ import annotations

from typing import List, Optional, Tuple, Dict
import pandas as pd
import numpy as np
from xgboost import XGBRegressor


# -------------------------
# Feature building
# -------------------------
def _add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["week_start"] = pd.to_datetime(out["week_start"])
    out = out.sort_values("week_start")
    out["weekofyear"] = out["week_start"].dt.isocalendar().week.astype(int)
    out["month"] = out["week_start"].dt.month.astype(int)
    out["year"] = out["week_start"].dt.year.astype(int)

    # numeric index (global time index)
    # IMPORTANT: caller should ensure ordering includes history then horizon
    out["t"] = np.arange(len(out), dtype=float)
    return out


def _prepare_variable_targets(
    cat_hist_df: pd.DataFrame,
    recurring_proj_df: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """
    cat_hist_df expected columns:
      - week_start
      - category_normalized
      - inflow_total
      - outflow_total

    recurring_proj_df expected columns (if provided):
      - week_start
      - category_normalized (or category)
      - outflow_total  (recurring outflow)

    Returns history with VARIABLE targets:
      variable_inflow_total  = inflow_total (v1 ignores recurring inflow)
      variable_outflow_total = max(0, outflow_total - recurring_outflow)
    """
    hist = cat_hist_df.copy()
    hist["week_start"] = pd.to_datetime(hist["week_start"])

    # always create the column so later code is safe
    hist["recurring_outflow"] = 0.0

    if recurring_proj_df is not None and not getattr(recurring_proj_df, "empty", True):
        rec = recurring_proj_df.copy()
        rec["week_start"] = pd.to_datetime(rec["week_start"])

        cat_col = "category_normalized" if "category_normalized" in rec.columns else "category"
        rec = rec.rename(columns={cat_col: "category_normalized"})

        # sum recurring outflow by (week, category)
        rec_grp = (
            rec.groupby(["week_start", "category_normalized"])["outflow_total"]
            .sum()
            .reset_index()
            .rename(columns={"outflow_total": "recurring_outflow"})
        )

        hist = hist.merge(
            rec_grp,
            on=["week_start", "category_normalized"],
            how="left",
            suffixes=("", "_r"),
        )

        # after merge, column may exist as recurring_outflow (from rec_grp) and/or original
        # unify safely:
        if "recurring_outflow_r" in hist.columns:
            hist["recurring_outflow"] = hist["recurring_outflow_r"].fillna(hist["recurring_outflow"])
            hist = hist.drop(columns=["recurring_outflow_r"])

        hist["recurring_outflow"] = hist["recurring_outflow"].fillna(0.0).astype(float)

    hist["variable_inflow_total"] = hist["inflow_total"].fillna(0.0).astype(float)
    hist["variable_outflow_total"] = (
        hist["outflow_total"].fillna(0.0).astype(float) - hist["recurring_outflow"].astype(float)
    ).clip(lower=0.0)

    return hist


def _fit_pooled_models(train_X: pd.DataFrame, y_in: np.ndarray, y_out: np.ndarray) -> Tuple[XGBRegressor, XGBRegressor]:
    params = dict(
        n_estimators=350,
        max_depth=4,
        learning_rate=0.06,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        random_state=42,
        objective="reg:squarederror",
    )
    m_in = XGBRegressor(**params)
    m_out = XGBRegressor(**params)
    m_in.fit(train_X.values, y_in)
    m_out.fit(train_X.values, y_out)
    return m_in, m_out


def _baseline_lastk(cat_hist: pd.DataFrame, k: int = 4) -> Tuple[float, float]:
    last = cat_hist.tail(k)
    if last.empty:
        return 0.0, 0.0
    return float(last["variable_inflow_total"].mean()), float(last["variable_outflow_total"].mean())


# -------------------------
# Main API
# -------------------------
def forecast_category_variable_xgb(
    cat_hist_df: pd.DataFrame,
    horizon_weeks: List[pd.Timestamp],
    recurring_proj_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Returns VARIABLE forecasts by category for the horizon.

    Output columns:
      - week_start
      - category_normalized
      - inflow_total   (variable inflow)
      - outflow_total  (variable outflow)
      - method         ("xgboost" or "baseline_fallback")
    """
    hist_var = _prepare_variable_targets(cat_hist_df, recurring_proj_df=recurring_proj_df)

    req = {"week_start", "category_normalized", "variable_inflow_total", "variable_outflow_total"}
    missing = req - set(hist_var.columns)
    if missing:
        raise ValueError(f"cat_hist_df missing required columns for xgb: {sorted(missing)}")

    horizon_weeks = [pd.to_datetime(w) for w in horizon_weeks]
    cats = sorted(hist_var["category_normalized"].dropna().unique().tolist())

    # ---------- Build pooled training table (week, category) ----------
    train_rows = hist_var[["week_start", "category_normalized", "variable_inflow_total", "variable_outflow_total"]].copy()
    train_rows = train_rows.dropna(subset=["category_normalized"])
    train_rows["category_normalized"] = train_rows["category_normalized"].astype(str)

    # Create combined timeline so "t" continues consistently into horizon
    # We build time features on ALL unique weeks (history + horizon)
    all_weeks = pd.Series(sorted(set(train_rows["week_start"].tolist() + horizon_weeks)))
    timeline = pd.DataFrame({"week_start": all_weeks})
    timeline = _add_time_features(timeline)

    # join time features into train_rows
    train_rows = train_rows.merge(
        timeline[["week_start", "t", "weekofyear", "month", "year"]],
        on="week_start",
        how="left",
    )

    # one-hot category on train
    X_train = train_rows[["t", "weekofyear", "month", "year", "category_normalized"]].copy()
    X_train = pd.get_dummies(X_train, columns=["category_normalized"], prefix="cat", dtype=float)

    y_in = train_rows["variable_inflow_total"].values.astype(float)
    y_out = train_rows["variable_outflow_total"].values.astype(float)

    # Guardrail: if overall history is too tiny, force baseline
    if len(train_rows) < 20:
        rows = []
        for cat in cats:
            cat_hist = hist_var[hist_var["category_normalized"] == cat].sort_values("week_start")
            inflow_base, outflow_base = _baseline_lastk(cat_hist, k=4)
            for ws in horizon_weeks:
                rows.append({
                    "week_start": ws,
                    "category_normalized": cat,
                    "inflow_total": inflow_base,
                    "outflow_total": outflow_base,
                    "method": "baseline_fallback",
                })
        return pd.DataFrame(rows)

    # Fit pooled models
    m_in, m_out = _fit_pooled_models(X_train, y_in, y_out)

    # ---------- Build horizon design matrix (week, category) ----------
    horizon_rows = []
    for ws in horizon_weeks:
        for cat in cats:
            horizon_rows.append({"week_start": ws, "category_normalized": str(cat)})

    hdf = pd.DataFrame(horizon_rows)
    hdf = hdf.merge(
        timeline[["week_start", "t", "weekofyear", "month", "year"]],
        on="week_start",
        how="left",
    )

    X_h = hdf[["t", "weekofyear", "month", "year", "category_normalized"]].copy()
    X_h = pd.get_dummies(X_h, columns=["category_normalized"], prefix="cat", dtype=float)

    # ✅ CRITICAL FIX: force horizon to have the exact same columns as training
    X_h = X_h.reindex(columns=X_train.columns, fill_value=0.0)

    pin = np.clip(m_in.predict(X_h.values).astype(float), 0.0, None)
    pout = np.clip(m_out.predict(X_h.values).astype(float), 0.0, None)

    # ---------- Per-category fallback rule ----------
    # If a category has too few history points, use baseline for that category only.
    hist_counts = (
        hist_var.groupby("category_normalized")["week_start"]
        .nunique()
        .to_dict()
    )

    rows = []
    idx = 0
    for ws in horizon_weeks:
        for cat in cats:
            n_pts = int(hist_counts.get(cat, 0))
            if n_pts < 6:
                cat_hist = hist_var[hist_var["category_normalized"] == cat].sort_values("week_start")
                inflow_base, outflow_base = _baseline_lastk(cat_hist, k=4)
                rows.append({
                    "week_start": ws,
                    "category_normalized": cat,
                    "inflow_total": float(inflow_base),
                    "outflow_total": float(outflow_base),
                    "method": "baseline_fallback",
                })
            else:
                rows.append({
                    "week_start": ws,
                    "category_normalized": cat,
                    "inflow_total": float(pin[idx]),
                    "outflow_total": float(pout[idx]),
                    "method": "xgboost",
                })
            idx += 1

    out = pd.DataFrame(rows)
    out["week_start"] = pd.to_datetime(out["week_start"])
    return out
