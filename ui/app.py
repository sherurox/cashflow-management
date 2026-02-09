import os
import json
import requests
import streamlit as st
import pandas as pd
import plotly.express as px
import re

# ---- Formatting helpers ----
def fmt_currency(x):
    try:
        if x is None:
            return "—"
        return f"${float(x):,.2f}"
    except Exception:
        return "—"


def fmt_number(x):
    try:
        if x is None:
            return "—"
        return f"{float(x):,.2f}"
    except Exception:
        return "—"


# ---- Text/QA helpers ----
_ZERO_WIDTH_RE = re.compile(r"[\u200b-\u200f\u202a-\u202e\ufeff]")


def normalize_answer_text(text: str) -> str:
    """Clean Gemini answers that sometimes contain zero-width chars or per-character newlines."""
    if not text:
        return ""

    t = _ZERO_WIDTH_RE.sub("", str(text))

    # Heuristic: if most lines are <= 1 char, the text likely got split per-character.
    if "\n" in t:
        lines = t.splitlines()
        if lines:
            short = sum(1 for ln in lines if len(ln.strip()) <= 1)
            if short / max(len(lines), 1) > 0.6:
                t = "".join(lines)
            else:
                t = "\n".join(ln.strip() for ln in lines)

    # Normalize whitespace around punctuation
    t = re.sub(r"\s+,", ",", t)
    t = re.sub(r",\s*", ", ", t)
    t = re.sub(r"\s+\.", ".", t)
    t = re.sub(r"\s+", " ", t).strip()

    return t


def parse_week_totals_text(text: str) -> dict:
    """Extract the JSON dict from a `week_totals` chunk text, if present."""
    if not text:
        return {}
    s = str(text)
    i = s.find("{")
    if i == -1:
        return {}
    try:
        return json.loads(s[i:])
    except Exception:
        return {}


st.set_page_config(page_title="Cashflow QC Demo", layout="wide")

# ---- Config ----
DEFAULT_API_BASE = os.getenv("API_BASE_URL", "http://127.0.0.1:8000").rstrip("/")

st.title("Cashflow QC Demo UI")
st.caption(
    "Upload CSV → Ingest (with starting balance) → Run QC → View QC outputs from Mongo/Forecast → Test Q&A"
)

api_base = st.sidebar.text_input("API Base URL", value=DEFAULT_API_BASE).rstrip("/")
st.sidebar.write("Using:", api_base)

# ---- Helpers ----
def api_post_ingest(file_bytes: bytes, filename: str, starting_balance: float):
    url = f"{api_base}/cashflow/ingest"
    files = {"file": (filename, file_bytes, "text/csv")}
    data = {"starting_balance": str(float(starting_balance))}
    r = requests.post(url, files=files, data=data, timeout=120)
    r.raise_for_status()
    return r.json()

def api_get_status(ingestion_id: str):
    url = f"{api_base}/cashflow/ingest/{ingestion_id}/status"
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    return r.json()

def api_get_forecast(ingestion_id: str):
    url = f"{api_base}/cashflow/forecast"
    r = requests.get(url, params={"ingestion_id": ingestion_id}, timeout=120)
    r.raise_for_status()
    return r.json()

def api_post_qa(ingestion_id: str, question: str):
    url = f"{api_base}/cashflow/qa"
    payload = {"ingestion_id": ingestion_id, "question": question}
    r = requests.post(url, json=payload, timeout=180)
    r.raise_for_status()
    return r.json()

# ---- Session state ----
if "ingestion_id" not in st.session_state:
    st.session_state.ingestion_id = ""
if "status" not in st.session_state:
    st.session_state.status = None
if "forecast" not in st.session_state:
    st.session_state.forecast = None
if "qa_checks" not in st.session_state:
    st.session_state.qa_checks = {}
if "starting_balance_input" not in st.session_state:
    st.session_state.starting_balance_input = ""

# ---- Layout ----
left, right = st.columns([1, 2], gap="large")

with left:
    st.subheader("1) Upload & Ingest")
    uploaded = st.file_uploader("Upload bank transactions CSV", type=["csv"])

    st.markdown("**Starting balance (required)**")
    st.session_state.starting_balance_input = st.text_input(
        "Enter starting bank balance",
        value=st.session_state.starting_balance_input,
        placeholder="e.g., 150000",
        label_visibility="collapsed",
    )

    colA, colB = st.columns(2)
    with colA:
        ingest_clicked = st.button("Ingest CSV", use_container_width=True)
    with colB:
        qc_clicked = st.button("Run QC (Load 3 outputs)", use_container_width=True)

    st.divider()
    st.subheader("Current ingestion")
    st.write("**ingestion_id:**", st.session_state.ingestion_id if st.session_state.ingestion_id else "—")

    if ingest_clicked:
        if not uploaded:
            st.error("Please upload a CSV first.")
        else:
            sb_raw = (st.session_state.starting_balance_input or "").strip()
            if sb_raw == "":
                st.error("Starting balance is required.")
            else:
                try:
                    starting_balance = float(sb_raw)
                except Exception:
                    st.error("Starting balance must be a valid number (e.g., 150000).")
                else:
                    try:
                        resp = api_post_ingest(uploaded.getvalue(), uploaded.name, starting_balance)
                        st.session_state.ingestion_id = resp.get("ingestion_id", "")
                        st.success(f"Ingested successfully. ingestion_id = {st.session_state.ingestion_id}")

                        # fetch status
                        if st.session_state.ingestion_id:
                            st.session_state.status = api_get_status(st.session_state.ingestion_id)

                        # clear previous outputs
                        st.session_state.forecast = None
                        st.session_state.qa_checks = {}

                    except requests.RequestException as e:
                        st.error(f"Ingest failed: {e}")

    if st.session_state.ingestion_id:
        if st.button("Refresh Status", use_container_width=True):
            try:
                st.session_state.status = api_get_status(st.session_state.ingestion_id)
            except requests.RequestException as e:
                st.error(f"Status fetch failed: {e}")

    if st.session_state.status:
        st.markdown("**Ingestion status**")
        st.json(st.session_state.status)

    if qc_clicked:
        if not st.session_state.ingestion_id:
            st.error("Ingest a CSV first to get an ingestion_id.")
        else:
            try:
                st.session_state.forecast = api_get_forecast(st.session_state.ingestion_id)
                st.session_state.qa_checks = (st.session_state.forecast or {}).get("qa_checks", {}) or {}
                st.success("QC outputs loaded.")
            except requests.RequestException as e:
                st.error(f"QC load failed: {e}")

with right:
    st.subheader("2) QC Outputs (Validation Metrics)")


    if not st.session_state.forecast:
        st.info("Upload a CSV, ingest it with a starting balance, then click **Run QC (Load 3 outputs)**.")
    else:
        qc = st.session_state.qa_checks or {}

        # Required QC fields
        hist_min = qc.get("historical_min_balance")
        hist_min_date = qc.get("historical_min_balance_date")

        fc_min = qc.get("forecast_min_balance")
        fc_min_week = qc.get("forecast_min_balance_week")

        warn_max = qc.get("warning_outflow_gt_hist_max")
        warn_p95 = qc.get("warning_outflow_gt_hist_p95")

        hist_max = qc.get("historical_weekly_outflow_max")
        hist_p95 = qc.get("historical_weekly_outflow_p95")
        fc_max = qc.get("forecast_weekly_outflow_max")
        fc_max_week = qc.get("forecast_weekly_outflow_max_week")

        # ---- Display block 1: Historical min balance ----
        st.markdown("### Historical min balance")
        c1, c2 = st.columns(2)
        with c1:
            st.metric("historical_min_balance", fmt_currency(hist_min))
        with c2:
            st.metric("date", hist_min_date if hist_min_date else "—")

        st.divider()

        # ---- Display block 2: Forecast min balance ----
        st.markdown("### Forecast min balance")
        c1, c2 = st.columns(2)
        with c1:
            st.metric("forecast_min_balance", fmt_currency(fc_min))
        with c2:
            st.metric("week", fc_min_week if fc_min_week else "—")

        st.divider()

        # ---- Display block 3: Outflow anomaly warning ----
        st.markdown("### Outflow anomaly warning")
        if warn_max is True or warn_p95 is True:
            st.warning("Forecast weekly outflow exceeds historical max and/or p95. Review assumptions.")
        else:
            st.success("Forecast weekly outflow is within historical range (max/p95).")

        c1, c2 = st.columns(2)
        with c1:
            st.write("**warning_outflow_gt_hist_max:**", bool(warn_max) if warn_max is not None else "—")
            st.write("**historical_weekly_outflow_max:**", fmt_currency(hist_max))
        with c2:
            st.write("**warning_outflow_gt_hist_p95:**", bool(warn_p95) if warn_p95 is not None else "—")
            st.write("**historical_weekly_outflow_p95:**", fmt_currency(hist_p95))

        st.write("---")
        st.write("**forecast_weekly_outflow_max:**", fmt_currency(fc_max))
        st.write("**forecast_weekly_outflow_max_week:**", fc_max_week if fc_max_week else "—")



with st.expander("Details: Forecast charts & totals table", expanded=False):

    weeks = (st.session_state.forecast or {}).get("weeks", []) or []
    df = pd.DataFrame(weeks)

    if df.empty:
        st.info("No weekly forecast data available.")
    else:
        df["week_start"] = pd.to_datetime(df["week_start"])

        # Ensure new payables/deferral fields exist (backward compatible)
        for col in ["outflow_total_paid", "deferred_outflow_total", "payables_queue_balance_end"]:
            if col not in df.columns:
                df[col] = 0.0
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

        # ---- Deferral indicators ----
        df = df.sort_values("week_start")
        df["deferral_happened"] = df["deferred_outflow_total"].astype(float) > 0
        deferral_weeks = int(df["deferral_happened"].sum())

        max_queue = float(df["payables_queue_balance_end"].max()) if not df.empty else 0.0
        max_queue_row = df.loc[df["payables_queue_balance_end"].idxmax()] if max_queue > 0 else None
        max_queue_week = (
            pd.Timestamp(max_queue_row["week_start"]).strftime("%Y-%m-%d") if max_queue_row is not None else "—"
        )

        k1, k2, k3 = st.columns(3)
        k1.metric("Weeks with deferral", f"{deferral_weeks} / {len(df)}")
        k2.metric("Max payables queue", fmt_currency(max_queue))
        k3.metric("Max queue week", max_queue_week)

        if deferral_weeks > 0:
            st.info("Deferral occurred in at least one forecast week (paid outflow < total outflow).")
        else:
            st.success("No deferral occurred in this forecast horizon.")

        c1, c2 = st.columns(2, gap="large")

        with c1:
            st.markdown("**Ending balance (anchored)**")
            fig = px.line(df, x="week_start", y="ending_balance")
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            st.markdown("**Payables queue (end balance)**")
            fig = px.line(df, x="week_start", y="payables_queue_balance_end")
            st.plotly_chart(fig, use_container_width=True)

        c3, c4 = st.columns(2, gap="large")

        with c3:
            st.markdown("**Inflow vs Outflow (weekly)**")
            df_melt = df.melt(
                id_vars=["week_start"],
                value_vars=["inflow_total", "outflow_total"],
                var_name="metric",
                value_name="value",
            )
            fig = px.bar(
                df_melt,
                x="week_start",
                y="value",
                color="metric",
                barmode="group",
            )
            st.plotly_chart(fig, use_container_width=True)

        with c4:
            st.markdown("**Paid vs Deferred Outflow (weekly)**")
            df_pd = df[["week_start", "outflow_total_paid", "deferred_outflow_total"]].copy()
            df_pd = df_pd.melt(
                id_vars=["week_start"],
                value_vars=["outflow_total_paid", "deferred_outflow_total"],
                var_name="metric",
                value_name="value",
            )
            fig = px.bar(
                df_pd,
                x="week_start",
                y="value",
                color="metric",
                barmode="stack",
            )
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("**Net cash (weekly)**")
        fig = px.bar(df, x="week_start", y="net_cash")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("**Forecast totals (table)**")
        show_cols = [
            "week_start",
            "inflow_total",
            "outflow_total",
            "outflow_total_paid",
            "deferred_outflow_total",
            "payables_queue_balance_end",
            "net_cash",
            "ending_balance",
        ]

        df_show = df[show_cols].copy()
        # Format as currency strings for readability
        for c in [
            "inflow_total",
            "outflow_total",
            "outflow_total_paid",
            "deferred_outflow_total",
            "payables_queue_balance_end",
            "net_cash",
            "ending_balance",
        ]:
            df_show[c] = df_show[c].apply(fmt_currency)

        df_show["week_start"] = df_show["week_start"].dt.strftime("%Y-%m-%d")

        st.dataframe(df_show, use_container_width=True, height=320)


st.divider()
st.subheader("3) Q&A (Grounded)")

if not st.session_state.ingestion_id:
    st.info("Ingest a CSV first.")
else:
    question = st.text_input(
        "Question",
        value="What is the forecast_min_balance and which week does it occur? Also, is forecast outflow higher than historical max/p95?"
    )
    if st.button("Ask", use_container_width=True):
        try:
            qa = api_post_qa(st.session_state.ingestion_id, question)
            st.markdown("**Answer**")
            raw_answer = qa.get("answer", "") or ""
            st.write(normalize_answer_text(raw_answer) or "—")

            # Quick traceability hint
            supp = qa.get("supporting_records", [])
            if isinstance(supp, list) and supp:
                md0 = (supp[0] or {}).get("metadata", {}) or {}
                st.caption(
                    f"Top supporting record: type={md0.get('type','—')} week_start={md0.get('week_start','—')}"
                )

                # Source-of-truth deferral indicators (prefer week_totals record over LLM formatting)
                if md0.get("type") == "week_totals":
                    week_obj = parse_week_totals_text(md0.get("text") or "")
                    if week_obj:
                        paid = week_obj.get("outflow_total_paid")
                        deferred = week_obj.get("deferred_outflow_total")
                        queue = week_obj.get("payables_queue_balance_end")
                        total = week_obj.get("outflow_total")

                        st.markdown("**Deferral indicators (from week_totals)**")
                        d1, d2, d3, d4 = st.columns(4)
                        d1.metric("outflow_total", fmt_currency(total))
                        d2.metric("outflow_total_paid", fmt_currency(paid))
                        d3.metric("deferred_outflow_total", fmt_currency(deferred))
                        d4.metric("payables_queue_end", fmt_currency(queue))

                        try:
                            if float(deferred or 0.0) > 0:
                                st.warning("Deferral happened this week: some outflow was queued (paid < total).")
                            else:
                                st.success("No deferral this week: paid outflow equals total outflow.")
                        except Exception:
                            pass

                # If the top supporting record is a week_totals chunk and includes deferral fields, surface it
                if md0.get("type") == "week_totals":
                    txt0 = (md0.get("text") or "")
                    if "deferred_outflow_total" in txt0 and "\"deferred_outflow_total\": 0.0" not in txt0:
                        st.warning("This week includes deferred outflow (payables were queued due to insufficient cash).")

            st.markdown("**Supporting records (traceability)**")
            if isinstance(supp, list) and supp:
                st.dataframe(pd.DataFrame(supp), use_container_width=True, height=260)
            else:
                st.write(supp)

        except requests.RequestException as e:
            st.error(f"Q&A failed: {e}")
