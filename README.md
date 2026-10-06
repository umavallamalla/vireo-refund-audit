
# Vireo Refund Intelligence & Audit

## What this is

A small AI-assisted support-ticket audit tool for Vireo Audio.

It reconciles the supplied helpdesk and legacy support exports, normalizes the legacy money representation, removes migration duplicates, calculates refund trends, and flags two policy-review populations:

1. `GW-OTHER` refunds above the Rs 500 goodwill cap.
2. Tickets where a refund and replacement are both recorded.

The app also provides an AI-assisted review of a selected flagged ticket using the Google Gemini API. The model returns structured JSON containing a policy flag, issue type, reason-code consistency, confidence, and explanation. Financial totals and deterministic policy flags are calculated with Python/Pandas rather than by an LLM.

## Data assumptions

The supplied support policy says:
- Legacy/current system records can use different monetary representations.
- The legacy Freshdesk data must be interpreted using its native representation.
- GW-OTHER goodwill credits are capped at Rs 500 per ticket and require Team Lead approval.
- A customer should not receive both a refund and a replacement for the same order.
- Replacement planning cost is product unit cost plus Rs 340 for reverse pickup and forward shipping.

For duplicate `ticket_id` values present in both systems, the current `helpdesk` record is preferred. The supplied duplicate pairs were field-consistent apart from the legacy money representation.

## Run locally

Put the supplied CSV files into:

```text
data/
  tickets.csv
  agents.csv
  orders.csv
  products.csv
  customers.csv
```

Then:

```bash
python -m venv .venv
source .venv/bin/activate        # macOS/Linux
# Windows: .venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

Or set:

```bash
export VIREO_DATA_DIR=/path/to/data
streamlit run app.py
```

## Key audit results from the supplied data

- Raw ticket rows: 12,238
- Unique ticket IDs: 11,600
- Duplicate migration rows: 638
- Normalized refund value: Rs 67,09,932
- Refund tickets: 2,340
- GW-OTHER over Rs 500: 879 tickets / Rs 28,71,632
- Refund + replacement: 166 tickets
- Refund value on refund+replacement cases: Rs 5,74,191
- Replacement planning cost on those cases: Rs 2,96,970

These are review populations, not proof of misconduct or recoverable leakage.

## Validation approach

The audit was cross-checked against:
- ticket counts before/after deduplication
- source-system duplication
- legacy money normalization
- reason-code totals
- quarterly totals
- agent-level totals
- policy-rule populations

## AI use

AI is intentionally limited to text interpretation of individual flagged tickets.

Provider:
- Google Gemini API
- Python SDK: `google-genai`

The model is asked to return structured JSON containing:
- `policy_issue`
- `issue_type`
- `reason_code_consistent`
- `confidence`
- `explanation`

The model is not used to calculate financial totals or replace the deterministic policy checks. Python/Pandas remains the source of truth for financial reconciliation and policy-rule populations.

The API key is supplied through the `GEMINI_API_KEY` environment variable and is not stored in the source code.
