# 🎓 College Graduation Rate Explorer

Ask plain-English questions about U.S. college graduation rates and get back an
answer, a chart, and a table — powered by the Claude API translating natural
language into SQL over public [IPEDS](https://nces.ed.gov/ipeds/) data.

🌐 [Click here to try the app.](https://college-graduation-rate-explorer.streamlit.app/)

> **Prototype / proof of concept.** Built on public IPEDS data to demonstrate a
> natural-language-to-SQL data app. Verify any figure against the official source
> before institutional use.

---

## What it does

Type a question like:

- *"What's MIT's 6-year graduation rate in 2023?"*
- *"Compare graduation rates for Pell and non-Pell students at Cornell University in 2023"*
- *"Show a 10-year trend of Yale's graduation rate compared to other Ivy League schools"*
- *"How do Penn State's 4-year and 6-year graduation rates compare over time?"*
- *"Compare graduation rates by race and ethnicity at UCLA in 2023"*

…and the app identifies the institution(s), writes and runs the SQL, and returns a
plain-language summary alongside a chart and a downloadable table.

**Highlights**

- **Confirm-before-execute** — ambiguous institution names (e.g. "Cornell" →
  Cornell University vs. Cornell College) prompt a "Did you mean…?" confirmation
  before any query runs.
- **Conversational follow-ups** — "what about at Yale?" or "make the y-axis 80–100"
  are interpreted in the context of the previous few exchanges.
- **Adjustable charts** — change titles, axis labels/ranges, line colors and styles,
  bar sort order, and label rotation by asking in plain English.
- **Honest about missing data** — IPEDS suppresses small-cohort cells; the app
  reports "not reported" rather than inventing a number, and declines
  cross-institution averages it can't compute correctly.

---

## How it works

```
User question
     │
     ▼
┌─────────────────────────────────────────────────────────────┐
│ Claude API — three stages (different models for speed/quality)│
│   1. Parse   → intent + institution(s), rewrite follow-ups    │
│   2. Generate SQL → a single validated SELECT                 │
│   3. Summarize    → streamed plain-language answer            │
└─────────────────────────────────────────────────────────────┘
     │                    │
     ▼                    ▼
Institution           SQLite database
resolution         (read-only, validated
(verified against    single-SELECT only)
the real DB)              │
                          ▼
                 Streamlit UI: summary + Altair chart + table + CSV
```

Generated SQL is validated (single read-only `SELECT`, no data-modifying
keywords) and executed against a read-only connection, so a model never modifies
the database. Institution names the model proposes are verified against the
actual data before use, so a hallucinated name can't reach the query.

**Stack:** R (data prep) → SQLite → Claude API → Streamlit → Streamlit Community Cloud

---

## Data

- **Source:** [IPEDS Graduation Rate (GR) component](https://nces.ed.gov/ipeds/),
  U.S. Department of Education (public data).
- **Scope:** 4-year U.S. institutions (`ICLEVEL = 1`), collection years 2015–2024.
- **Metrics:** bachelor's completion within 4, 5, and 6 years; the overall
  graduation rate within 150% of normal time; broken out by sex, race/ethnicity,
  and Pell/loan status. The unqualified "graduation rate" means the **6-year
  bachelor's rate** (`GBA6RTT`).

**Caveats baked into the app**

- **Suppressed cells:** IPEDS suppresses small-*n* subgroups; missing values are
  reported as "not reported," never zero.
- **No cross-institution averages:** the database holds institution-level rates
  without cohort counts, so an unweighted average of rates would differ from the
  student-weighted figures official sources publish. The app declines these and
  suggests comparing specific institutions instead. (Ranking and filtering across
  institutions — highest/lowest/top-N — is supported.)
- **Reporting year ≠ entry year:** `year` is the collection year; the underlying
  cohort entered college roughly six years earlier.

`schema_context.txt` documents every column and the example question→SQL pairs
that ground the model. `codebook_gradrate.csv` maps raw IPEDS variable names to
readable labels.

---

## Run it locally

Requires Python 3.9+ and an [Anthropic API key](https://console.anthropic.com/).

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Provide your API key (environment variable)
export ANTHROPIC_API_KEY="sk-ant-..."      # Windows PowerShell: $env:ANTHROPIC_API_KEY="sk-ant-..."

# 3. Launch
streamlit run app.py
```

The app reads `Gradrate_150pct_2015_2024_4yr_inst.db` and `schema_context.txt`
from the repo root — no external database setup needed.

---

## Deploy on Streamlit Community Cloud

1. Push this repo to GitHub.
2. Create a new app at [share.streamlit.io](https://share.streamlit.io) pointing
   at `app.py`.
3. In **Settings → Secrets**, add your key:
   ```toml
   ANTHROPIC_API_KEY = "sk-ant-..."
   ```
   This is stored securely and is **never** committed to the repo.

Every API call incurs Anthropic usage costs on the key you configure.

---

## Project structure

| File | Purpose |
|---|---|
| `app.py` | The Streamlit app and the full NL→SQL pipeline |
| `schema_context.txt` | Column docs, metric rules, and example question→SQL pairs given to the model |
| `codebook_gradrate.csv` | IPEDS variable name → readable label reference |
| `Gradrate_150pct_2015_2024_4yr_inst.db` | The SQLite database the app queries |
| `.streamlit/config.toml` | Theme (colors, fonts) |
| `requirements.txt` | Python dependencies |

---

## Limitations

This is a demonstration, not a validated reporting tool. It covers a single narrow
slice of IPEDS data, is subject to the caveats above, and — like any LLM-driven
system — can misinterpret an ambiguous question. Always confirm figures against
the [official IPEDS data](https://nces.ed.gov/ipeds/) before relying on them.

---

## License & data

Application code is provided as-is for demonstration purposes. The underlying
graduation-rate data is public and sourced from IPEDS (U.S. Department of
Education, National Center for Education Statistics).
