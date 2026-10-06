
import os
import json
import pandas as pd
import numpy as np
import streamlit as st

try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None
    types = None

st.set_page_config(page_title="Vireo Refund Intelligence", layout="wide")

# Gemini configuration. The API key is read from the GEMINI_API_KEY
# environment variable and is never hard-coded into the app.
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
gemini_client = None
if genai is not None and GEMINI_API_KEY:
    try:
        gemini_client = genai.Client(api_key=GEMINI_API_KEY)
    except Exception:
        gemini_client = None

DATA_DIR = os.getenv("VIREO_DATA_DIR", "data")

@st.cache_data
def load_data(data_dir):
    tickets = pd.read_csv(os.path.join(data_dir, "tickets.csv"))
    agents = pd.read_csv(os.path.join(data_dir, "agents.csv"))
    products = pd.read_csv(os.path.join(data_dir, "products.csv"))

    for col in ["created_at", "first_response_at", "resolved_at"]:
        tickets[col] = pd.to_datetime(tickets[col], errors="coerce")

    tickets["refund_norm"] = tickets["refund_amount_inr"].fillna(0)
    tickets.loc[tickets["source_system"].eq("legacy_fd"), "refund_norm"] /= 100

    # Migration duplicates: prefer the current helpdesk record when both systems
    # contain the same ticket_id. The duplicate pairs were field-consistent
    # apart from the legacy currency representation.
    tickets["source_rank"] = tickets["source_system"].map(
        {"helpdesk": 0, "legacy_fd": 1}
    ).fillna(9)

    clean = (
        tickets.sort_values(["ticket_id", "source_rank"])
        .drop_duplicates("ticket_id", keep="first")
        .copy()
    )

    clean["quarter"] = clean["created_at"].dt.to_period("Q").astype(str)
    clean["month"] = clean["created_at"].dt.to_period("M").astype(str)

    clean = clean.merge(
        agents[["agent_id", "name", "team", "tier", "site", "shift"]],
        on="agent_id",
        how="left",
    )
    clean = clean.merge(
        products[["sku", "product_name", "family", "unit_cost_inr"]],
        left_on="product_sku",
        right_on="sku",
        how="left",
        suffixes=("", "_product"),
    )

    clean["gw_over_500"] = (
        clean["refund_reason_code"].eq("GW-OTHER")
        & clean["refund_norm"].gt(500)
    )
    clean["refund_plus_replacement"] = (
        clean["refund_norm"].gt(0)
        & clean["replacement_issued"].eq("Y")
    )
    clean["replacement_planning_cost"] = np.where(
        clean["replacement_issued"].eq("Y"),
        clean["unit_cost_inr"].fillna(0) + 340,
        0,
    )
    return clean


def ai_review_prompt(row):
    return f"""
You are reviewing one Vireo Audio support ticket for policy exceptions.

Return ONLY valid JSON with these keys:
policy_issue (boolean)
issue_type (string)
reason_code_consistent (boolean or null)
confidence (number from 0 to 1)
explanation (string)

Ticket ID: {row.ticket_id}
Refund amount (INR): {row.refund_norm}
Refund reason code: {row.refund_reason_code}
Replacement issued: {row.replacement_issued}
Category: {row.category}
Assigned team: {row.assigned_team}
Customer message: {row.customer_message}
Agent notes: {row.agent_notes}

Policy facts:
- GW-OTHER goodwill credits are capped at Rs 500 per ticket and require Team Lead approval.
- A customer should not receive both a refund and a replacement for the same order.
- DOA within 7 days permits a refund or replacement.
- WTY-BUYBACK is warranty buy-back.
Do not invent facts that are not present in the ticket.
""".strip()


st.title("Vireo Refund Intelligence & Audit")
st.caption("Deterministic financial reconciliation + policy exception review")

try:
    df = load_data(DATA_DIR)
except Exception as e:
    st.error(f"Could not load data from '{DATA_DIR}': {e}")
    st.stop()

refunds = df[df["refund_norm"] > 0].copy()

# Sidebar
st.sidebar.header("Filters")
teams = ["All"] + sorted(df["team"].dropna().unique().tolist())
team = st.sidebar.selectbox("Team", teams)
reasons = ["All"] + sorted(refunds["refund_reason_code"].dropna().unique().tolist())
reason = st.sidebar.selectbox("Refund reason", reasons)

filtered = df.copy()
if team != "All":
    filtered = filtered[filtered["team"] == team]
if reason != "All":
    filtered = filtered[filtered["refund_reason_code"] == reason]

fref = filtered[filtered["refund_norm"] > 0]

# KPI cards
c1, c2, c3, c4 = st.columns(4)
c1.metric("Unique tickets", f"{len(filtered):,}")
c2.metric("Refund tickets", f"{len(fref):,}")
c3.metric("Refund value", f"₹{fref.refund_norm.sum()/100000:.2f}L")
c4.metric(
    "Policy flags",
    f"{int(filtered.gw_over_500.sum() + filtered.refund_plus_replacement.sum()):,}",
)

st.divider()

tab1, tab2, tab3, tab4 = st.tabs(
    ["Executive Summary", "Refund Analysis", "Agent Analysis", "Policy Audit"]
)

with tab1:
    st.subheader("Monthly refund trend")
    monthly = (
        fref.groupby("month")
        .agg(refund_amount_inr=("refund_norm", "sum"), refund_tickets=("ticket_id", "size"))
        .reset_index()
    )
    st.line_chart(monthly.set_index("month")["refund_amount_inr"])

    st.subheader("Quarter comparison")
    quarterly = (
        filtered.groupby("quarter")
        .agg(
            tickets=("ticket_id", "size"),
            refund_tickets=("refund_norm", lambda x: (x > 0).sum()),
            refund_amount_inr=("refund_norm", "sum"),
            csat_responses=("csat_score", lambda x: x.notna().sum()),
            csat_mean=("csat_score", "mean"),
        )
        .reset_index()
    )
    quarterly["refund_rate"] = quarterly["refund_tickets"] / quarterly["tickets"]
    quarterly["avg_refund_inr"] = (
        quarterly["refund_amount_inr"] / quarterly["refund_tickets"]
    )
    st.dataframe(
        quarterly.style.format(
            {
                "refund_amount_inr": "₹{:,.0f}",
                "refund_rate": "{:.1%}",
                "avg_refund_inr": "₹{:,.0f}",
                "csat_mean": "{:.2f}",
            }
        ),
        use_container_width=True,
    )

with tab2:
    st.subheader("Refund reasons")
    reason_summary = (
        fref.groupby("refund_reason_code")
        .agg(
            refund_tickets=("ticket_id", "size"),
            refund_amount_inr=("refund_norm", "sum"),
        )
        .reset_index()
        .sort_values("refund_amount_inr", ascending=False)
    )
    st.bar_chart(reason_summary.set_index("refund_reason_code")["refund_amount_inr"])
    st.dataframe(
        reason_summary.style.format({"refund_amount_inr": "₹{:,.0f}"}),
        use_container_width=True,
    )

with tab3:
    st.subheader("Agent refund performance")
    agent_summary = (
        filtered.groupby(["agent_id", "name", "team", "tier"])
        .agg(
            tickets=("ticket_id", "size"),
            refund_tickets=("refund_norm", lambda x: (x > 0).sum()),
            refund_amount_inr=("refund_norm", "sum"),
        )
        .reset_index()
    )
    agent_summary["refund_rate"] = (
        agent_summary["refund_tickets"] / agent_summary["tickets"]
    )
    agent_summary["avg_refund_inr"] = (
        agent_summary["refund_amount_inr"]
        / agent_summary["refund_tickets"].replace(0, np.nan)
    )
    st.dataframe(
        agent_summary.sort_values("refund_amount_inr", ascending=False).style.format(
            {
                "refund_amount_inr": "₹{:,.0f}",
                "refund_rate": "{:.1%}",
                "avg_refund_inr": "₹{:,.0f}",
            }
        ),
        use_container_width=True,
    )

with tab4:
    st.subheader("Potential policy exceptions")

    a, b, c = st.columns(3)
    a.metric(
        "GW-OTHER > ₹500",
        f"{int(filtered.gw_over_500.sum()):,}",
        f"₹{filtered.loc[filtered.gw_over_500, 'refund_norm'].sum()/100000:.2f}L",
    )
    b.metric(
        "Refund + replacement",
        f"{int(filtered.refund_plus_replacement.sum()):,}",
        f"₹{filtered.loc[filtered.refund_plus_replacement, 'refund_norm'].sum()/100000:.2f}L refund",
    )
    c.metric(
        "Replacement planning cost",
        f"₹{filtered.loc[filtered.refund_plus_replacement, 'replacement_planning_cost'].sum()/100000:.2f}L",
    )

    st.warning(
        "These are review flags, not automatic findings of misconduct. "
        "Approval records or legitimate exceptions may exist outside the supplied CSVs."
    )

    flags = filtered[
        filtered.gw_over_500 | filtered.refund_plus_replacement
    ].copy()

    show_cols = [
        "ticket_id", "created_at", "agent_id", "name", "team",
        "refund_reason_code", "refund_norm", "replacement_issued",
        "gw_over_500", "refund_plus_replacement", "customer_message",
        "agent_notes",
    ]
    st.dataframe(flags[show_cols].sort_values("refund_norm", ascending=False),
                 use_container_width=True)

    st.subheader("AI-assisted ticket review")
    if len(flags):
        selected = st.selectbox("Select a flagged ticket", flags.ticket_id.tolist())
        row = flags[flags.ticket_id.eq(selected)].iloc[0]
        st.write(
            f"**₹{row.refund_norm:,.0f}** | "
            f"{row.refund_reason_code} | "
            f"replacement={row.replacement_issued}"
        )
        prompt = ai_review_prompt(row)

        st.markdown("**AI review prompt**")
        st.code(prompt, language="text")

        if gemini_client is None:
            if genai is None:
                st.error(
                    "Gemini SDK is not installed. Run: "
                    "`python3 -m pip install -U google-genai`"
                )
            elif not GEMINI_API_KEY:
                st.warning(
                    "Gemini API key is not configured. Set the GEMINI_API_KEY "
                    "environment variable and restart Streamlit."
                )
            else:
                st.error("Gemini client could not be initialized.")
        else:
            if st.button("Run AI Review", type="primary"):
                with st.spinner("Gemini is reviewing this ticket..."):
                    try:
                        # Try the current Flash model first, then fall back to
                        # other stable Flash models if Google temporarily has
                        # insufficient capacity (503 UNAVAILABLE).
                        models_to_try = [
                            "gemini-3.8-flash",
                            "gemini-3.7-flash",
                            "gemini-3.5-flash",
                            "gemini-3.5-flash-lite",
                        ]
                        response = None
                        last_error = None

                        for model_name in models_to_try:
                            try:
                                response = gemini_client.models.generate_content(
                                    model=model_name,
                                    contents=prompt,
                                    config=types.GenerateContentConfig(
                                        response_mime_type="application/json",
                                        temperature=0.1,
                                    ),
                                )
                                break
                            except Exception as model_error:
                                last_error = model_error
                                if "503" not in str(model_error) and "UNAVAILABLE" not in str(model_error):
                                    raise

                        if response is None:
                            raise last_error

                        result = json.loads(response.text)

                        required = {
                            "policy_issue",
                            "issue_type",
                            "reason_code_consistent",
                            "confidence",
                            "explanation",
                        }
                        missing = required - set(result.keys())

                        if missing:
                            st.error(
                                "Gemini returned incomplete JSON. Missing: "
                                + ", ".join(sorted(missing))
                            )
                        else:
                            st.success("AI review completed")
                            r1, r2, r3 = st.columns(3)
                            r1.metric(
                                "Policy issue",
                                "Yes" if result["policy_issue"] else "No",
                            )
                            r2.metric(
                                "Reason-code consistent",
                                (
                                    "Yes"
                                    if result["reason_code_consistent"] is True
                                    else "No"
                                    if result["reason_code_consistent"] is False
                                    else "Unknown"
                                ),
                            )
                            r3.metric(
                                "Confidence",
                                f"{float(result['confidence']):.0%}",
                            )

                            st.markdown(
                                f"**Issue type:** {result['issue_type']}"
                            )
                            st.markdown(
                                f"**Explanation:** {result['explanation']}"
                            )

                            with st.expander("Raw AI JSON"):
                                st.json(result)

                            st.caption(
                                "AI review is interpretive. Deterministic policy "
                                "checks and financial totals above remain the source "
                                "of truth."
                            )

                    except json.JSONDecodeError:
                        st.error(
                            "Gemini did not return valid JSON. Raw response:"
                        )
                        st.code(response.text, language="text")
                    except Exception as e:
                        st.error(f"AI review failed: {e}")
