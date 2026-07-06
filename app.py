"""IPEDS Graduation Rate Explorer.

Streamlit app: users ask natural-language questions about 6-year graduation
rates at 4-year institutions (IPEDS 2015-2024); Claude parses intent,
generates SQL against the local SQLite database, and summarizes results.

Pipeline:
  1. Parse   — Claude extracts institution mentions + intent (structured output).
  2. Resolve — mentions are matched against actual INSTNM values in the DB;
               ambiguous matches trigger a confirm-before-execute form
               (state held in st.session_state.pending).
  3. SQL     — Claude generates a single SELECT using the schema doc + resolved
               names; validated as read-only before execution.
  4. Answer  — results table + chart + Claude-written plain-language summary.

Run:  streamlit run app.py   (requires ANTHROPIC_API_KEY in the environment)
"""

import json
import math
import os
import re
import sqlite3
from pathlib import Path
from typing import List, Literal, Optional

import altair as alt
import anthropic
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from pydantic import BaseModel

APP_DIR = Path(__file__).parent
DB_PATH = APP_DIR / "Gradrate_150pct_2015_2024_4yr_inst.db"
SCHEMA_PATH = APP_DIR / "schema_context.txt"
# Per-step models: fast models for the light structured/prose steps, the strongest
# model kept for SQL correctness. Change these to re-balance speed vs. quality.
MODEL_PARSE = "claude-haiku-4-5-20251001"  # intent + follow-up rewrite + name extraction
MODEL_SQL = "claude-opus-4-8"              # SQL generation — correctness matters most
MODEL_SUMMARY = "claude-sonnet-4-6"        # plain-language result summary (streamed)
MODEL_CHART = "claude-opus-4-8"            # chart-spec edits (unchanged)

MAX_AMBIGUOUS_OPTIONS = 20  # cap options shown in a confirmation dropdown
# Cap rows sent to the summarizer. Matches the SQL stage's LIMIT 200 so the summary
# normally sees the entire result set; only pathological over-limit results are sampled.
MAX_SUMMARY_ROWS = 200

EXAMPLE_QUESTIONS = [
    "Show a 10-year trend of Yale's graduation rate compared to other Ivy League schools",
    "Compare graduation rates for Pell and non-Pell students at Cornell University in 2023",
    "What's MIT's 6-year graduation rate in 2023?",
    "How do Penn State's 4-year and 6-year graduation rates compare over time?",
    "Compare graduation rates for men and women at the University of Michigan over the last 10 years",
    "Compare graduation rates by race and ethnicity at UCLA in 2023",
]


# ---------------------------------------------------------------------------
# Structured-output models (Claude fills these via client.messages.parse)
# ---------------------------------------------------------------------------

class Mention(BaseModel):
    """One institution or group the user referred to."""
    text: str                    # the name as the user wrote it (no possessives)
    official_names: List[str]    # exact IPEDS INSTNM values if confidently known, else []


class ParsedQuestion(BaseModel):
    intent: Literal["data_query", "chart_adjustment", "out_of_scope"]
    standalone_question: str     # the message rewritten to be self-contained using prior context
    mentions: List[Mention]
    out_of_scope_reply: Optional[str]


class SeriesStyle(BaseModel):
    """Per-series visual overrides, matched against displayed series values."""
    series: str                   # exact series value as shown (e.g. 'Yale University')
    color: Optional[str]          # CSS color name or hex, e.g. 'darkblue' / '#00356b'
    dash: Optional[Literal["solid", "dashed", "dotted"]]
    marker: Optional[Literal["circle", "square", "triangle", "diamond", "cross"]]


class SQLPlan(BaseModel):
    sql: Optional[str]           # single SELECT statement, or null if declining
    decline_reason: Optional[str]
    chart_type: Literal["line", "bar", "none"]
    x_column: Optional[str]
    y_columns: List[str]
    series_column: Optional[str]  # e.g. INSTNM for multi-institution trend lines
    y_min: Optional[float]        # optional y-axis bounds (rates are 0-100)
    y_max: Optional[float]
    title: Optional[str]          # null -> no title (user can add one via follow-up)
    x_label: Optional[str]        # null -> default from the display-name mapping
    y_label: Optional[str]        # null -> default from the display-name mapping
    series_styles: List[SeriesStyle]   # usually [] unless the question requests styling
    default_color: Optional[str]  # color for series WITHOUT an explicit style entry
    show_table: bool              # false unless the user explicitly asks to see the table
    x_label_angle: Optional[Literal[0, 45, 90]]  # x-axis label rotation; null = default
    sort_order: Optional[Literal["ascending", "descending", "alphabetical"]]  # bar order; null = default


class ChartSpec(BaseModel):
    """Standalone chart spec, used when the user adjusts an existing chart."""
    chart_type: Literal["line", "bar", "none"]
    x_column: Optional[str]
    y_columns: List[str]
    series_column: Optional[str]
    y_min: Optional[float]
    y_max: Optional[float]
    title: Optional[str]
    x_label: Optional[str]
    y_label: Optional[str]
    series_styles: List[SeriesStyle]
    default_color: Optional[str]
    show_table: bool
    x_label_angle: Optional[Literal[0, 45, 90]]  # x-axis label rotation; null = default
    sort_order: Optional[Literal["ascending", "descending", "alphabetical"]]  # bar order; null = default
    # True ONLY when the user's requested change maps to NO field above (e.g. gridlines,
    # fonts, 3D). Lets the app say "I can't adjust that yet" instead of silently no-op'ing.
    unsupported_request: bool


# ---------------------------------------------------------------------------
# Cached resources
# ---------------------------------------------------------------------------

@st.cache_resource
def get_client() -> anthropic.Anthropic:
    return anthropic.Anthropic()


@st.cache_data
def load_schema_context() -> str:
    return SCHEMA_PATH.read_text(encoding="utf-8")


@st.cache_data
def display_names() -> dict[str, str]:
    """Column -> readable label, parsed from the 'Display name mapping' section of
    schema_context.txt (only mapping lines have the `- NAME: "Label"` shape)."""
    return dict(re.findall(r'^- (\w+): "(.+)"$', load_schema_context(), re.MULTILINE))


@st.cache_data
def distinct_institutions() -> list[str]:
    with _db_connection() as con:
        rows = con.execute(
            "SELECT DISTINCT INSTNM FROM grad_rates WHERE ICLEVEL = 1 ORDER BY INSTNM"
        ).fetchall()
    return [r[0] for r in rows]


def _db_connection() -> sqlite3.Connection:
    """Read-only connection so generated SQL can never modify the database."""
    uri = DB_PATH.resolve().as_uri() + "?mode=ro"
    return sqlite3.connect(uri, uri=True)


# ---------------------------------------------------------------------------
# Conversation context (recent exchanges fed back into the parse stage)
# ---------------------------------------------------------------------------

MAX_CONTEXT_EXCHANGES = 3


def recent_context(max_exchanges: int = MAX_CONTEXT_EXCHANGES) -> str:
    """Compact transcript of the last few Q&A exchanges for follow-up resolution."""
    exchanges: list[tuple[str, str]] = []
    current_q: Optional[str] = None
    for item in st.session_state.history:
        if item["role"] == "user":
            current_q = item["text"]
        elif current_q is not None:
            if item.get("kind") == "result":
                reply = (
                    f"[ran a query] summary: {item['summary'][:500]}\n"
                    f"SQL: {item['sql']}\n"
                    f"chart spec: {json.dumps(item['chart'])}"
                )
            else:
                reply = item.get("text", "")[:300]
            exchanges.append((current_q, reply))
            current_q = None
    if not exchanges:
        return "(no prior conversation)"
    return "\n\n".join(
        f"User: {q}\nAssistant: {a}" for q, a in exchanges[-max_exchanges:]
    )


def last_result_item() -> Optional[dict]:
    for item in reversed(st.session_state.history):
        if item["role"] == "assistant" and item.get("kind") == "result":
            return item
    return None


# ---------------------------------------------------------------------------
# Stage 1 — parse the question with Claude
# ---------------------------------------------------------------------------

def parse_question(question: str, context: str) -> ParsedQuestion:
    system = f"""You are the question-parsing stage of an app that answers questions about \
6-year graduation rates at 4-year U.S. institutions (IPEDS collection years 2015-2024).

Reference — database schema, notes, and named institution groups:

{load_schema_context()}

The user message includes the recent conversation. Use it ONLY to resolve references in \
the NEW message (e.g. "what about at Yale?" after a Pell question means Pell rates at Yale; \
"make the y-axis 80-100" refers to the most recent chart). If the new message is \
self-contained or about an unrelated topic, IGNORE the prior conversation entirely — never \
carry institutions, years, or subgroups into a question that doesn't reference them.

Your job in THIS stage (do NOT write SQL here):
1. Set intent:
   - "data_query": needs data from the grad_rates table (rates, subgroups, rankings, \
trends, comparisons — including follow-ups that need a NEW or modified query).
   - "chart_adjustment": the message only changes how the MOST RECENT result is displayed \
(axis ranges, line vs bar, which columns are plotted, colors/styles, or showing/hiding the \
data table — "show me the table" / "hide the table") and needs NO new data. If different \
data is needed (new institution, year, or metric), that is "data_query", not \
"chart_adjustment".
   - "out_of_scope": not answerable from this table (admissions, tuition, SAT scores, \
enrollment, 2-year colleges...) OR a cross-institution aggregate/average that the schema's \
"aggregate questions must be declined" rule prohibits (any "national/overall/statewide/average \
rate," or averaging across multiple institutions). Write out_of_scope_reply. For a prohibited \
aggregate, use the exact decline explanation given in that schema rule. Ranking/filtering \
across institutions (highest/lowest/top-N) is still a valid data_query, NOT out_of_scope.
2. standalone_question: rewrite the new message as one fully self-contained request, \
folding in whatever prior context it references (institutions, subgroups, years, metrics, \
chart being adjusted). If the message is already self-contained, return it unchanged. \
The downstream SQL stage sees ONLY this rewrite, so it must not depend on the conversation.
3. mentions: every institution or named group referenced by the standalone_question. \
For each mention:
   - text: the institution name as written, minus possessives ("Cornell's" -> "Cornell").
   - official_names: exact official IPEDS INSTNM values, ONLY if you are confident \
(e.g. "MIT" -> ["Massachusetts Institute of Technology"]). Expand named groups \
(e.g. "Ivy League") into all member official names. If unsure of the exact official name, \
leave official_names EMPTY — the app matches the text against the database itself. Never guess.
   For chart_adjustment, mentions can be empty (no new data is fetched).

Questions with no specific institution (state or national rankings, averages) are valid \
data queries with an empty mentions list."""

    user_msg = f"""Recent conversation (oldest first):
{context}

NEW message from the user: {question}"""

    response = get_client().messages.parse(
        model=MODEL_PARSE,
        max_tokens=4096,
        thinking={"type": "disabled"},  # Haiku 4.5 predates adaptive thinking
        # System prompt is identical on every parse call -> cache the schema block.
        system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": user_msg}],
        output_format=ParsedQuestion,
    )
    if response.stop_reason == "refusal" or response.parsed_output is None:
        raise RuntimeError("The model declined to parse this question. Try rephrasing it.")
    return response.parsed_output


# ---------------------------------------------------------------------------
# Stage 2 — resolve mentions against real INSTNM values (local, deterministic)
# ---------------------------------------------------------------------------

# Common institution shorthands -> exact IPEDS INSTNM. Multi-campus universities and
# acronyms otherwise resolve to nothing (e.g. "UCLA" is not a substring of "University
# of California-Los Angeles") or to many campuses ("University of Michigan" matches Ann
# Arbor/Dearborn/Flint), which would fire the ambiguity confirmation. Mapping the common
# shorthand to its flagship campus makes these resolve deterministically — important for
# the one-click example questions. Matched case-insensitively against both the user's
# wording and any official name the model proposes. Extend as needed.
INSTITUTION_ALIASES = {
    "mit": "Massachusetts Institute of Technology",
    "ucla": "University of California-Los Angeles",
    "penn state": "Pennsylvania State University-Main Campus",
    "penn state university": "Pennsylvania State University-Main Campus",
    "pennsylvania state university": "Pennsylvania State University-Main Campus",
    "university of michigan": "University of Michigan-Ann Arbor",
    "umich": "University of Michigan-Ann Arbor",
}


def resolve_mention(mention: Mention) -> tuple[str, list[str]]:
    """Return (status, names) where status is 'resolved' | 'ambiguous' | 'not_found'.

    Claude's suggested official names are trusted only if they exist verbatim in the
    database; otherwise we fall back to substring matching on the user's own words,
    so a hallucinated name can never reach the SQL stage.
    """
    all_names = distinct_institutions()
    by_lower = {n.lower(): n for n in all_names}

    # 0. Known shorthand -> exact flagship campus (deterministic; see INSTITUTION_ALIASES).
    #    Checked against the user's wording first, then any model-proposed official name.
    for candidate in [mention.text, *mention.official_names]:
        alias = INSTITUTION_ALIASES.get((candidate or "").strip().lower())
        if alias and alias.lower() in by_lower:
            return "resolved", [by_lower[alias.lower()]]

    # 1. Claude-suggested official names, verified against the DB
    if mention.official_names:
        verified = [by_lower[n.lower()] for n in mention.official_names if n.lower() in by_lower]
        if verified and len(verified) == len(mention.official_names):
            return "resolved", verified
        # partial/failed verification -> fall through to fuzzy matching

    text = mention.text.strip()
    if not text:
        return "not_found", []

    # 2. Exact (case-insensitive) match on the user's own text
    if text.lower() in by_lower:
        return "resolved", [by_lower[text.lower()]]

    # 3. Substring match
    matches = [n for n in all_names if text.lower() in n.lower()]
    if len(matches) == 1:
        return "resolved", matches
    if len(matches) > 1:
        return "ambiguous", matches[:MAX_AMBIGUOUS_OPTIONS]
    return "not_found", []


# ---------------------------------------------------------------------------
# Stage 3 — generate SQL with Claude
# ---------------------------------------------------------------------------

def generate_sql(question: str, resolved_names: list[str]) -> SQLPlan:
    system = f"""You translate questions about college graduation rates into a single \
SQLite SELECT statement against the grad_rates table.

{load_schema_context()}

Rules:
- Output exactly ONE SELECT statement. No semicolons, no comments, no INSERT/UPDATE/DDL.
- ALWAYS include ICLEVEL = 1 in the WHERE clause.
- When the question is about specific institutions, use ONLY the exact names listed under \
"Resolved institutions" in the user message — never invent, alter, or add names. If the \
question requires an institution but none is resolved, set sql to null with a decline_reason.
- If the question cannot be answered from this table, set sql to null and explain in \
decline_reason.
- NEVER compute a cross-institution average/aggregate (no AVG/SUM of a rate across multiple \
institutions, no "national/overall/statewide/average rate"). Per the schema's "aggregate \
questions must be declined" rule, set sql to null and put that rule's exact decline text in \
decline_reason. Ranking/filtering across institutions (ORDER BY ... LIMIT for highest/lowest/ \
top-N) is allowed. Aggregating over YEARS for a SINGLE institution is also allowed.
- Exclude NULL rates when ranking or ordering (add IS NOT NULL).
- Cap open-ended result sets with LIMIT 200.

Chart spec (for the app to render):
- Trend over years: chart_type "line", x_column "year". With multiple institutions, return \
long-format rows (INSTNM, year, rate), set series_column to "INSTNM", and put the rate \
column in y_columns.
- Comparison across institutions or categories in one year: chart_type "bar" with \
x_column "INSTNM" (or return a single row with several rate columns in y_columns and \
x_column null — the app will transpose it).
- Single number or wide lookup: chart_type "none".
- x_column, y_columns, and series_column must reference columns actually selected in the SQL.
- Leave title, x_label, and y_label null unless the question itself asks for specific labeling \
— the app defaults to the schema's display-name mapping. Likewise leave series_styles as [] \
and default_color null unless the question explicitly requests colors or line styles.
- show_table: false unless the question explicitly asks to see the table / raw data / rows. \
(The app always offers a CSV download, and automatically shows the table when there is no \
chart, so false is the right default.)
- x_label_angle (0/45/90) and sort_order (ascending/descending/alphabetical for bar charts): \
leave both null unless the question explicitly asks to rotate the x-axis labels or order the \
bars by value."""

    user_msg = f"""Question: {question}

Resolved institutions (exact INSTNM values verified against the database):
{chr(10).join('- ' + n for n in resolved_names) if resolved_names else '(none — question is not institution-specific)'}"""

    response = get_client().messages.parse(
        model=MODEL_SQL,
        max_tokens=16000,
        thinking={"type": "adaptive"},
        # System prompt is identical on every SQL call -> cache the schema block.
        system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": user_msg}],
        output_format=SQLPlan,
    )
    if response.stop_reason == "refusal" or response.parsed_output is None:
        raise RuntimeError("The model declined to generate SQL for this question.")
    return response.parsed_output


def adjust_chart(request: str, last_result: dict) -> tuple[dict, bool]:
    """Update the most recent chart's spec per the user's request. No new data is
    fetched. Returns (new_chart_dict, unsupported) — unsupported is True when the
    request maps to no adjustable property, so the caller can say so explicitly."""
    df = last_result["df"]
    long_df, _, _ = to_long_form(last_result["chart"], df)
    series_values = list(dict.fromkeys(long_df["Series"])) if long_df is not None else []

    system = f"""You update a chart specification for an existing query result. The user \
wants to change how the chart is DISPLAYED — do not invent new data or columns.

Available dataframe columns: {list(df.columns)}
Series currently plotted (exact values to use in series_styles): {json.dumps(series_values)}
Current chart spec: {json.dumps(last_result["chart"])}
Display-name mapping (used automatically when a label field is null): \
{json.dumps({k: v for k, v in display_names().items() if k in df.columns})}

Rules:
- Change ONLY what the user asked for; copy every other field from the current spec \
(including any existing series_styles entries the user didn't mention).
- x_column, y_columns, and series_column must be columns from the list above.
- Rates are whole-number percentages, so y_min/y_max are on a 0-100 scale \
(e.g. "y-axis 80-100%" -> y_min 80, y_max 100). null means the default axis: bounds \
rounded to multiples of 10 padded ~10 points beyond the data (upper capped at 100, lower \
floored at 0) — so only set these when the user asks for a specific range \
("reset the y-axis" -> both null).
- title / x_label / y_label: set to the user's requested text when they ask for a title \
or axis label; null means "use the default" (no title; axis labels from the mapping above). \
To REMOVE a custom title or label, set the field back to null.
- Per-series styling: series_styles entries must use the exact series values listed above. \
color is a CSS color name or hex; dash (solid/dashed/dotted) and marker \
(circle/square/triangle/diamond/cross) apply to line charts. \
"Make Yale dark blue and everyone else gray" -> one entry for Yale University with color \
"darkblue", plus default_color "gray". default_color applies to every series without its \
own entry; null means the standard palette. Unspecified fields inside an entry stay null. \
To clear styling, return series_styles [] and default_color null.
- show_table: set true when the user asks to see the table / raw data ("show me the table"), \
false when they ask to hide it; otherwise copy the current value.
- x_label_angle: rotate the x-axis labels. 0 = horizontal, 45 = diagonal, 90 = vertical \
("make the x-axis labels horizontal" -> 0). null = default. Applies mainly to bar charts.
- sort_order (bar charts): order the bars. "descending" = highest value on the left, \
"ascending" = lowest first, "alphabetical" = by category name (the reset/default). \
"order the bars from highest to lowest" -> "descending". null = default order.
- unsupported_request: set true ONLY when the user's requested change maps to NONE of the \
fields above (e.g. gridlines, fonts, 3D, annotations, background color, legend position). \
If ANY part of the request is expressible, set false and apply that part. When true, still \
return a valid spec (copy the current one unchanged)."""

    response = get_client().messages.parse(
        model=MODEL_CHART,
        max_tokens=8000,
        thinking={"type": "adaptive"},
        system=system,
        messages=[{"role": "user", "content": request}],
        output_format=ChartSpec,
    )
    if response.stop_reason == "refusal" or response.parsed_output is None:
        raise RuntimeError("The model declined to adjust the chart.")
    spec = response.parsed_output
    new_chart = {
        "type": spec.chart_type, "x": spec.x_column, "y": spec.y_columns,
        "series": spec.series_column, "y_min": spec.y_min, "y_max": spec.y_max,
        "title": spec.title, "x_label": spec.x_label, "y_label": spec.y_label,
        "series_styles": [s.model_dump() for s in spec.series_styles],
        "default_color": spec.default_color, "show_table": spec.show_table,
        "x_label_angle": spec.x_label_angle, "sort_order": spec.sort_order,
    }
    # Treat as unsupported if the model flagged it, or if nothing actually changed
    # (a defensive fallback so a no-op adjustment never looks like a silent success).
    unsupported = spec.unsupported_request or (new_chart == last_result["chart"])
    return new_chart, unsupported


# ---------------------------------------------------------------------------
# SQL validation + execution
# ---------------------------------------------------------------------------

_FORBIDDEN = re.compile(
    r"\b(insert|update|delete|drop|alter|create|attach|detach|pragma|vacuum|replace|reindex)\b",
    re.IGNORECASE,
)


def validate_sql(sql: str) -> str:
    cleaned = sql.strip().rstrip(";").strip()
    if ";" in cleaned:
        raise ValueError("Multiple SQL statements are not allowed.")
    if not re.match(r"^\s*select\b", cleaned, re.IGNORECASE):
        raise ValueError("Only SELECT statements are allowed.")
    if _FORBIDDEN.search(cleaned):
        raise ValueError("Query contains a disallowed keyword.")
    return cleaned


def run_sql(sql: str) -> pd.DataFrame:
    with _db_connection() as con:
        return pd.read_sql_query(sql, con)


# ---------------------------------------------------------------------------
# Stage 4 — plain-language summary with Claude
# ---------------------------------------------------------------------------

def summary_preview(df: pd.DataFrame) -> pd.DataFrame:
    """Rows to show the summarizer. Normally the whole result (<= MAX_SUMMARY_ROWS).
    When larger, sample so EVERY institution/series is represented rather than blindly
    taking head() — otherwise ORDER BY INSTNM drops whole schools off the tail."""
    if len(df) <= MAX_SUMMARY_ROWS:
        return df
    group_col = next((c for c in ("INSTNM", "Series") if c in df.columns), None)
    if group_col is None:
        return df.head(MAX_SUMMARY_ROWS)
    groups = df.groupby(group_col, sort=False)
    if groups.ngroups > MAX_SUMMARY_ROWS:
        # more groups than the row budget — full coverage is impossible; a 2-5
        # sentence summary over that many institutions is general anyway.
        return df.head(MAX_SUMMARY_ROWS)
    per_group = max(1, MAX_SUMMARY_ROWS // groups.ngroups)
    sampled = groups.head(per_group)               # first N rows of each group
    if len(sampled) < MAX_SUMMARY_ROWS:            # backfill remaining budget in order
        extra = df.drop(sampled.index).head(MAX_SUMMARY_ROWS - len(sampled))
        sampled = pd.concat([sampled, extra])
    return sampled.sort_index()


def stream_summary(question: str, sql: str, df: pd.DataFrame):
    """Yield the plain-language summary as text deltas, for st.write_stream."""
    preview = summary_preview(df)
    system = """You write short plain-language answers about college graduation rate data.
- Rate columns are whole-number percentages: display 87 as 87%. Never multiply by 100.
- NULL/NaN means IPEDS suppressed the value (small cohort) — call it "not reported", never zero.
- The 'year' column is the IPEDS collection year, not the year students entered college.
- 2-5 sentences, direct and factual. Answer the question first, then any notable context
  (trends, gaps, missing data). If the table is empty, say no matching data was found."""
    note = (f", showing a representative sample of {len(preview)} covering every "
            f"institution/series" if len(preview) < len(df) else "")
    user_msg = f"""Question: {question}

SQL that was run: {sql}

Result ({len(df)} row(s){note}):
{preview.to_markdown(index=False)}"""

    with get_client().messages.stream(
        model=MODEL_SUMMARY,
        max_tokens=2000,
        thinking={"type": "disabled"},  # a short factual summary needs no thinking
        system=system,
        messages=[{"role": "user", "content": user_msg}],
    ) as stream:
        for text in stream.text_stream:
            yield text


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run_full_query(question: str, resolved_names: list[str],
                   interpreted_as: Optional[str] = None) -> dict:
    """SQL -> exec -> streamed summary. Renders the assistant bubble live (so the
    summary appears token-by-token) and returns the completed history item."""
    plan = generate_sql(question, resolved_names)

    if plan.sql is None:
        text = plan.decline_reason or "I can't answer that from the graduation-rate data."
        with st.chat_message("assistant"):
            st.write(text)
        return {"role": "assistant", "kind": "text", "text": text}

    sql = validate_sql(plan.sql)
    df = run_sql(sql)
    item = {
        "role": "assistant", "kind": "result",
        "summary": None, "sql": sql, "df": df,
        "interpreted_as": interpreted_as,
        "chart": {
            "type": plan.chart_type, "x": plan.x_column,
            "y": plan.y_columns, "series": plan.series_column,
            "y_min": plan.y_min, "y_max": plan.y_max,
            "title": plan.title, "x_label": plan.x_label, "y_label": plan.y_label,
            "series_styles": [s.model_dump() for s in plan.series_styles],
            "default_color": plan.default_color, "show_table": plan.show_table,
            "x_label_angle": plan.x_label_angle, "sort_order": plan.sort_order,
        },
    }
    # The new item's index once appended (used for a stable download-button key).
    idx = len(st.session_state.history)
    with st.chat_message("assistant"):
        if interpreted_as:
            st.caption(f"Interpreted as: *{interpreted_as}*")
        item["summary"] = st.write_stream(stream_summary(question, sql, df)) \
            or "(No summary generated — see the results below.)"
        render_result_visuals(item, idx)
    return item


def handle_new_question(question: str) -> None:
    """Parse + resolve. Either answers directly or parks a confirmation in session_state."""
    parsed = parse_question(question, recent_context())

    if parsed.intent == "out_of_scope":
        st.session_state.history.append({
            "role": "assistant", "kind": "text",
            "text": parsed.out_of_scope_reply
                    or "I can only answer questions about 6-year graduation rates "
                       "at 4-year institutions (2015-2024).",
        })
        return

    if parsed.intent == "chart_adjustment":
        last = last_result_item()
        if last is None:
            st.session_state.history.append({
                "role": "assistant", "kind": "text",
                "text": "There's no chart to adjust yet — ask a data question first.",
            })
            return
        new_chart, unsupported = adjust_chart(parsed.standalone_question, last)
        if unsupported:
            st.session_state.history.append({
                "role": "assistant", "kind": "text",
                "text": "I can't adjust that yet. I can change the chart title, axis labels, "
                        "y-axis range, line colors/styles/markers, x-axis label rotation, "
                        "bar sort order, and show/hide the data table.",
            })
            return
        st.session_state.history.append({
            "role": "assistant", "kind": "result",
            "summary": "Here's the updated chart.",
            "sql": last["sql"], "df": last["df"],
            "interpreted_as": None, "chart": new_chart,
        })
        return

    # Downstream stages see only the standalone rewrite, never raw history.
    question_final = parsed.standalone_question.strip() or question
    interpreted_as = question_final if question_final != question else None

    resolved: dict[str, list[str]] = {}
    ambiguous: dict[str, list[str]] = {}
    not_found: list[str] = []

    for mention in parsed.mentions:
        status, names = resolve_mention(mention)
        if status == "resolved":
            resolved[mention.text] = names
        elif status == "ambiguous":
            ambiguous[mention.text] = names
        else:
            not_found.append(mention.text)

    if not_found:
        st.session_state.history.append({
            "role": "assistant", "kind": "text",
            "text": "I couldn't find "
                    + ", ".join(f'"{t}"' for t in not_found)
                    + " among the 4-year institutions in this dataset. "
                      "Try the fuller official name (e.g. \"University of California-Berkeley\").",
        })
        return

    if ambiguous:
        # Confirm-before-execute: nothing runs until the user picks a match.
        st.session_state.pending = {
            "question": question_final, "interpreted_as": interpreted_as,
            "resolved": resolved, "ambiguous": ambiguous,
        }
        return

    all_names = [n for names in resolved.values() for n in names]
    st.session_state.history.append(
        run_full_query(question_final, all_names, interpreted_as)
    )


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

TABLEAU10 = ["#4c78a8", "#f58518", "#e45756", "#72b7b2", "#54a24b",
             "#eeca3b", "#b279a2", "#ff9da6", "#9d755d", "#bab0ac"]
DASH_PATTERNS = {"solid": [1, 0], "dashed": [8, 4], "dotted": [2, 2]}
MARKER_SHAPES = {"circle": "circle", "square": "square", "triangle": "triangle-up",
                 "diamond": "diamond", "cross": "cross"}
# Spec value (0/45/90) -> Vega labelAngle. Negative reads left-to-right/upward, which
# is the conventional readable orientation for rotated category labels.
LABEL_ANGLE = {0: 0, 45: -45, 90: -90}
# Spec sort_order -> Vega x-encoding sort. "-y"/"y" sort bars by their value.
BAR_SORT = {"descending": "-y", "ascending": "y", "alphabetical": "ascending"}


def shorten_series_labels(labels: list[str]) -> dict[str, str]:
    """When every label shares one prefix with a parenthetical qualifier — e.g.
    '6-Year Graduation Rate (Men)' / '(Women)' — keep just the qualifier so long
    legend entries don't get truncated to an identical prefix. A bare prefix
    ('6-Year Graduation Rate') becomes 'Total'. Otherwise labels are unchanged."""
    parsed, prefixes = {}, set()
    for label in labels:
        m = re.match(r"^(.+?) \((.+)\)$", label)
        if m:
            prefixes.add(m.group(1))
            parsed[label] = m.group(2)
        else:
            prefixes.add(label)
            parsed[label] = "Total"
    if len(labels) > 1 and len(prefixes) == 1:
        return parsed
    return {label: label for label in labels}


def to_long_form(chart: dict, df: pd.DataFrame):
    """Reshape a result frame to (long_df, x, y_cols) with columns Series/Rate.
    Series values carry display names (melted wide rows are raw column names);
    melted column labels are additionally shortened for readable legends."""
    x, y, series = chart.get("x"), chart.get("y") or [], chart.get("series")
    y_cols = [c for c in y if c in df.columns]
    melted = False
    if series and series in df.columns and x in df.columns and y_cols:
        long_df = df[[x, series, y_cols[0]]].rename(
            columns={series: "Series", y_cols[0]: "Rate"})
    elif x in df.columns and y_cols:
        long_df = df.melt(id_vars=[x], value_vars=y_cols,
                          var_name="Series", value_name="Rate")
        melted = True
    elif len(df) == 1 and y_cols:
        # single wide row (e.g. race subgroups) -> categories on the x-axis
        long_df = df[y_cols].melt(var_name="Series", value_name="Rate")
        x = "Series"
        melted = True
    else:
        return None, None, y_cols
    long_df = long_df.dropna(subset=["Rate"])
    dn = display_names()
    long_df["Series"] = long_df["Series"].map(lambda s: dn.get(str(s), str(s)))
    if melted:
        # only column-derived labels are shortened, never institution names
        short = shorten_series_labels(list(dict.fromkeys(long_df["Series"])))
        long_df["Series"] = long_df["Series"].map(short)
    return long_df, x, y_cols


def render_chart(item: dict) -> bool:
    """Render via Altair (per-series styling and axis domains need it).
    Returns True only if a chart was actually drawn."""
    chart, df = item["chart"], item["df"]
    if chart["type"] == "none" or df.empty:
        return False
    try:
        long_df, x, y_cols = to_long_form(chart, df)
        if long_df is None or long_df.empty:
            return False
        dn = display_names()
        series = chart.get("series")

        # Default y-axis for rates (bounds are always multiples of 10):
        #   upper = (max + 10) rounded UP to the next 10, capped at 100
        #   lower = (min - 10) rounded DOWN to the previous 10, floored at 0
        # e.g. data 87-96 -> 70-100. An explicit y_min/y_max (from the question
        # or a follow-up) always wins.
        y_min, y_max = chart.get("y_min"), chart.get("y_max")
        highest, lowest = float(long_df["Rate"].max()), float(long_df["Rate"].min())
        default_hi = min(100.0, math.ceil((highest + 10) / 10) * 10)
        default_lo = max(0.0, math.floor((lowest - 10) / 10) * 10)
        lo = y_min if y_min is not None else default_lo
        hi = y_max if y_max is not None else default_hi
        y_scale = alt.Scale(domain=[lo, hi], clamp=True)

        # Axis labels: explicit user request wins, else the display-name mapping
        x_label = chart.get("x_label") or ("" if x == "Series" else dn.get(x, x))
        if chart.get("y_label"):
            y_label = chart["y_label"]
        elif len(y_cols) == 1:
            y_label = dn.get(y_cols[0], y_cols[0])
        else:
            y_label = "Graduation Rate (%)"
        series_title = dn.get(series, series) if series else None

        # --- per-series styling (color / dash / marker) ---
        series_vals = list(dict.fromkeys(long_df["Series"]))
        styles = {s["series"].lower(): s for s in (chart.get("series_styles") or [])}
        default_color = chart.get("default_color")

        def style_of(val: str) -> dict:
            return styles.get(val.lower(), {})

        color_enc = alt.Color("Series:N", title=series_title)
        if styles or default_color:
            color_range = [
                style_of(s).get("color") or default_color or TABLEAU10[i % len(TABLEAU10)]
                for i, s in enumerate(series_vals)
            ]
            color_enc = alt.Color("Series:N", title=series_title,
                                  scale=alt.Scale(domain=series_vals, range=color_range))

        y_enc = alt.Y("Rate:Q", title=y_label, scale=y_scale)
        # x-axis: optional label rotation, plus value-sorting for bar charts.
        x_kwargs = {"title": x_label}
        angle = chart.get("x_label_angle")
        if angle in LABEL_ANGLE:
            x_kwargs["axis"] = alt.Axis(labelAngle=LABEL_ANGLE[angle])
        if chart["type"] == "bar" and chart.get("sort_order") in BAR_SORT:
            x_kwargs["sort"] = BAR_SORT[chart["sort_order"]]
        x_enc = alt.X(f"{x}:O", **x_kwargs)
        base = alt.Chart(long_df)

        if chart["type"] == "line":
            # NOTE: same-field channels (color/shape/strokeDash on Series) get their
            # legends MERGED by Vega-Lite. legend=None on any merged channel suppresses
            # the whole merged legend — so give every channel the same title instead,
            # producing one combined legend (color + dash + marker per series).
            line_enc = {"x": x_enc, "y": y_enc, "color": color_enc}
            if any(style_of(s).get("dash") for s in series_vals):
                dash_range = [DASH_PATTERNS[style_of(s).get("dash") or "solid"]
                              for s in series_vals]
                line_enc["strokeDash"] = alt.StrokeDash(
                    "Series:N", title=series_title,
                    scale=alt.Scale(domain=series_vals, range=dash_range))
            line = base.mark_line().encode(**line_enc)
            if any(style_of(s).get("marker") for s in series_vals):
                shape_range = [MARKER_SHAPES[style_of(s).get("marker") or "circle"]
                               for s in series_vals]
                points = base.mark_point(filled=True, size=70).encode(
                    x=x_enc, y=y_enc, color=color_enc,
                    shape=alt.Shape("Series:N", title=series_title,
                                    scale=alt.Scale(domain=series_vals, range=shape_range)))
            else:
                points = base.mark_point(filled=True, size=45).encode(
                    x=x_enc, y=y_enc, color=color_enc)
            c = line + points
        else:
            bar_color = (color_enc if (styles or default_color)
                         else alt.Color("Series:N", title=None, legend=None))
            c = base.mark_bar().encode(x=x_enc, y=y_enc, color=bar_color)
        if chart.get("title"):
            c = c.properties(title=chart["title"])
        st.altair_chart(c, width="stretch")
        return True
    except Exception:
        return False  # caller falls back to showing the table


TABLE_PREVIEW_ROWS = 5
# ~5 data rows + header visible; the rest reachable via the table's own scrollbar
TABLE_PREVIEW_HEIGHT = 38 + 35 * TABLE_PREVIEW_ROWS


def render_result_visuals(item: dict, idx: int) -> None:
    """Chart + CSV download + (optional) table. Shared by live streaming and replay;
    does NOT render the summary text (the caller handles that)."""
    chart_drawn = render_chart(item)
    display_df = item["df"].rename(columns=display_names())
    csv_bytes = display_df.to_csv(index=False).encode("utf-8")
    # Visual-first: the table appears only on explicit request ("show me the
    # table") or when there is no chart to look at.
    if item["chart"].get("show_table") or not chart_drawn:
        btn_col, table_col = st.columns([1, 4])
        with btn_col:
            st.download_button(
                "⬇ Download CSV", csv_bytes,
                file_name="grad_rates_results.csv", mime="text/csv",
                key=f"dl_{idx}", help=f"All {len(display_df)} rows",
                width="stretch",
            )
            st.caption(f"{len(display_df)} rows")
        with table_col:
            # full data in one scrollable table, sized to preview ~5 rows;
            # "content" (not None) is auto-height in streamlit >= 1.58
            st.dataframe(
                display_df, width="stretch",
                height=(TABLE_PREVIEW_HEIGHT
                        if len(display_df) > TABLE_PREVIEW_ROWS else "content"),
            )
    else:
        st.download_button(
            "⬇ Download CSV", csv_bytes,
            file_name="grad_rates_results.csv", mime="text/csv",
            key=f"dl_{idx}",
            help=f"All {len(display_df)} rows (ask 'show me the table' to view)",
        )
    with st.expander("Show SQL Query"):
        st.code(item["sql"], language="sql")


def render_item(item: dict, idx: int) -> None:
    if item["role"] == "user":
        with st.chat_message("user"):
            st.write(item["text"])
        return
    with st.chat_message("assistant"):
        if item["kind"] == "text":
            st.write(item["text"])
        elif item["kind"] == "result":
            if item.get("interpreted_as"):
                st.caption(f"Interpreted as: *{item['interpreted_as']}*")
            st.write(item["summary"])
            render_result_visuals(item, idx)


def render_confirmation_form() -> None:
    pending = st.session_state.pending
    with st.chat_message("assistant"):
        st.write("Before I run anything — a couple of names match more than one institution. "
                 "Which did you mean?")
        with st.form("confirm_form"):
            choices: dict[str, str] = {}
            for text, options in pending["ambiguous"].items():
                choices[text] = st.selectbox(
                    f'"{text}"', options + ["None of these"], key=f"choice_{text}",
                )
            col_run, col_cancel = st.columns(2)
            run = col_run.form_submit_button("Run query", type="primary")
            cancel = col_cancel.form_submit_button("Cancel")

    if cancel:
        st.session_state.pending = None
        st.session_state.history.append({
            "role": "assistant", "kind": "text", "text": "Okay, cancelled — nothing was run.",
        })
        st.rerun()

    if run:
        rejected = [t for t, c in choices.items() if c == "None of these"]
        if rejected:
            st.session_state.pending = None
            st.session_state.history.append({
                "role": "assistant", "kind": "text",
                "text": "No match confirmed for "
                        + ", ".join(f'"{t}"' for t in rejected)
                        + ". Try asking again with the institution's fuller official name.",
            })
            st.rerun()
        resolved = dict(pending["resolved"])
        for text, choice in choices.items():
            resolved[text] = [choice]
        question = pending["question"]
        interpreted_as = pending.get("interpreted_as")
        st.session_state.pending = None
        all_names = [n for names in resolved.values() for n in names]
        with st.spinner("Running query..."):
            try:
                st.session_state.history.append(
                    run_full_query(question, all_names, interpreted_as))
            except Exception as exc:  # surface API/SQL errors in the chat
                st.session_state.history.append({
                    "role": "assistant", "kind": "text", "text": f"Something went wrong: {exc}",
                })
        st.rerun()


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Graduation Rate Explorer", page_icon="🎓", layout="wide")

# Stop Ctrl/Cmd+C (copy) from triggering Streamlit's global "c" = Clear caches
# shortcut. The listener is registered on the parent WINDOW in the capture phase:
# capture runs window -> document -> target, so a window-capture listener fires
# before ANY document-level listener (which is where Streamlit registers its
# shortcut handler during mount), regardless of registration order. We only
# stopImmediatePropagation (not preventDefault), so the browser's native copy is
# untouched. Runs from the component iframe (same-origin: sandbox allows it).
components.html(
    """
    <script>
      (function () {
        const win = window.parent;
        if (win.__copyShortcutGuard) return;
        win.__copyShortcutGuard = true;
        const swallowCopy = function (e) {
          const k = (e.key || '').toLowerCase();
          if ((e.ctrlKey || e.metaKey) && (k === 'c' || e.keyCode === 67 || e.which === 67)) {
            e.stopImmediatePropagation();  // hide it from Streamlit's shortcut handler
          }
        };
        // Belt-and-suspenders: window first (beats any document listener), then
        // document, both in the capture phase.
        win.addEventListener('keydown', swallowCopy, true);
        win.document.addEventListener('keydown', swallowCopy, true);
      })();
    </script>
    """,
    height=0,
)

# Minimal identity CSS: serif title with a thin gold accent rule, and navy
# pill-shaped example buttons. Everything else inherits the config.toml theme.
st.markdown(
    """
    <style>
      /* Pin the title + caption to the top as the conversation scrolls.
         Sticky must sit on the keyed container's WRAPPER (its parent is the tall
         content block, so it can travel); sticky on the inner container fails
         because that container's parent is only header-height.
         top = Streamlit's fixed toolbar height (~3.75rem) so the TITLE pins just
         below the toolbar instead of hiding behind it (the toolbar is opaque and
         sits above this header). */
      [data-testid="stLayoutWrapper"]:has(> [class*="st-key-app-header"]) {
        position: sticky;
        top: 3.75rem;
        z-index: 100;
        background-color: #FAF8F5;   /* matches canvas so content scrolls under cleanly */
        padding-top: 0.5rem;
        padding-bottom: 0.35rem;
      }
      /* Cover the strip between the toolbar and the pinned header so scrolled
         content doesn't flash through the ~0.5rem gap above it. */
      [data-testid="stLayoutWrapper"]:has(> [class*="st-key-app-header"])::before {
        content: "";
        position: absolute;
        left: 0; right: 0;
        top: -0.75rem; height: 0.75rem;
        background-color: #FAF8F5;
      }
      /* !important overrides Streamlit's higher-specificity default heading CSS */
      h1.app-title {
        font-family: Georgia, 'Times New Roman', serif !important;
        font-weight: 600 !important;
        font-size: 2.1rem;
        color: #1F3A5F !important;
        margin: 0 0 0.35rem 0;
        padding-bottom: 0.4rem;
        border-bottom: 2px solid #B08A3E;   /* thin gold accent rule */
      }
      /* Example-question pills (scoped to the keyed sidebar container) */
      [class*="st-key-example-pills"] button {
        background-color: #1F3A5F;
        color: #FFFFFF;
        border: none;
        border-radius: 999px;
        padding: 0.45rem 1rem;
        text-align: left;
        font-size: 0.86rem;
        line-height: 1.3;
        transition: background-color 0.15s ease;
      }
      [class*="st-key-example-pills"] button:hover {
        background-color: #2C4E7A;
        color: #FFFFFF;
      }
      [class*="st-key-example-pills"] button:focus:not(:active) {
        color: #FFFFFF;
        box-shadow: 0 0 0 2px #B08A3E;
      }
    </style>
    """,
    unsafe_allow_html=True,
)
with st.container(key="app-header"):
    st.markdown('<h1 class="app-title">🎓 Graduation Rate Explorer</h1>',
                unsafe_allow_html=True)
    st.caption("Ask about 6-year graduation rates at 4-year U.S. institutions "
               "(IPEDS collection years 2015-2024).  \n"
               "Prototype — built on public IPEDS data. Verify results before institutional use.")

if not DB_PATH.exists():
    st.error(f"Database not found: {DB_PATH}")
    st.stop()

# Initialize state before the sidebar so example buttons can read `pending`.
if "history" not in st.session_state:
    st.session_state.history = []
if "pending" not in st.session_state:
    st.session_state.pending = None
if "example_click" not in st.session_state:
    st.session_state.example_click = None

if not os.environ.get("ANTHROPIC_API_KEY"):
    st.sidebar.warning("ANTHROPIC_API_KEY is not set — API calls will fail unless "
                       "credentials are available another way (e.g. `ant auth login`).")

with st.sidebar:
    if st.button("🧹 New topic (clear conversation)", width="stretch"):
        st.session_state.history = []
        st.session_state.pending = None
        st.rerun()
    st.subheader("Example questions")
    st.caption("Click to ask.")
    with st.container(key="example-pills"):
        for ex in EXAMPLE_QUESTIONS:
            if st.button(ex, key=f"ex_{hash(ex)}", width="stretch",
                         disabled=st.session_state.pending is not None):
                st.session_state.example_click = ex
                st.rerun()

chat_col = st.container()
with chat_col:
    for i, item in enumerate(st.session_state.history):
        render_item(item, i)
    if st.session_state.pending:
        render_confirmation_form()

question = st.chat_input(
    "Ask a question about graduation rates...",
    disabled=st.session_state.pending is not None,
)

# A clicked example pill feeds the same flow as typed input.
if st.session_state.example_click and not st.session_state.pending:
    question = st.session_state.example_click
    st.session_state.example_click = None

if question:
    st.session_state.history.append({"role": "user", "text": question})
    # Run inside chat_col so the live-streamed assistant bubble lands in the chat column.
    with chat_col:
        with st.chat_message("user"):
            st.write(question)
        with st.spinner("Thinking..."):
            try:
                handle_new_question(question)
            except anthropic.AuthenticationError:
                st.session_state.history.append({
                    "role": "assistant", "kind": "text",
                    "text": "Authentication failed — check that ANTHROPIC_API_KEY is set correctly.",
                })
            except Exception as exc:
                st.session_state.history.append({
                    "role": "assistant", "kind": "text", "text": f"Something went wrong: {exc}",
                })
    st.rerun()
