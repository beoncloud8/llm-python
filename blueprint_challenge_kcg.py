# Made By: Kevin Cliff Gunawan
# Patterns: Sequential Chain + Judge and Critic + Skills and Tools

import os
import json
import re
import asyncio
import requests
from typing import List, Literal
from dataclasses import dataclass
import pandas as pd
import matplotlib.pyplot as plt

from agents import Agent, Runner, function_tool, trace

from dotenv import load_dotenv
load_dotenv()

SECTORS_API_KEY = os.getenv('SECTORS_API_KEY')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

print(f"Debug: SECTORS_API_KEY loaded: {SECTORS_API_KEY is not None}")
print(f"Debug: OPENAI_API_KEY loaded: {OPENAI_API_KEY is not None}")

if not SECTORS_API_KEY or not OPENAI_API_KEY:
    print("Error: Please set SECTORS_API_KEY and OPENAI_API_KEY environment variables")
    print("You can set them in your shell or create a .env file")
    exit(1)

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

headers  = {"Authorization": SECTORS_API_KEY}
BASE_URL = "https://api.sectors.app"
print("Environment configured.")

### Step 1: Tools
def retrieve_from_endpoint(url: str) -> str:
    try:
        r = requests.get(url, headers=headers)
        r.raise_for_status()
    except requests.exceptions.HTTPError as err:
        print(f"    HTTP error: {err}")
        return json.dumps({"error": str(err)})
    data = r.json()
    if isinstance(data, dict) and "results" in data:
        count = len(data["results"])
        print(f"    {count} result(s) returned")
        if count == 0:
            note = data.get("llm_translation", {}).get("message", "")
            if note:
                print(f"    API note: {note}")
    return json.dumps(data)

@function_tool
def get_company_overview(ticker: str, country: str = "indonesia") -> str:
    """
    Get company overview from Singapore Exchange (SGX), Indonesia Exchange (IDX), or Malaysia (KLSE).

    :param ticker: Stock ticker symbol, e.g. 'BBCA' for IDX or 'D05' for SGX
    :param country: One of 'indonesia', 'singapore', or 'malaysia'. Default: 'indonesia'
    """
    assert country.lower() in ["indonesia", "singapore", "malaysia"], \
        "Country must be either Indonesia, Singapore, or Malaysia"

    if country.lower() == "indonesia":
        url = f"{BASE_URL}/v1/company/report/{ticker}/?sections=overview"
    elif country.lower() == "singapore":
        url = f"{BASE_URL}/v1/sgx/company/report/{ticker}/"
    elif country.lower() == "malaysia":
        url = f"{BASE_URL}/v1/klse/company/report/{ticker}/"

    try:
        return retrieve_from_endpoint(url)
    except Exception as e:
        print(f"Error occurred: {e}")
        return json.dumps({"error": str(e)})


@function_tool
def get_company_financials(ticker: str) -> str:
    """
    Fetch financial and dividend data for a single IDX company.
    Returns: revenue, earnings, EPS, ROE, ROA, PE, dividend yield per year.
    :param ticker: IDX ticker without .JK suffix, e.g. 'BBCA'
    """
    return retrieve_from_endpoint(
        f"{BASE_URL}/v1/company/report/{ticker}/?sections=financials,dividend"
    )


@function_tool
def get_top_companies_ranked(dimension: str) -> str:
    """
    Return a list of top companies (symbol) based on certain dimension
    (dividend yield, total dividend, revenue, earnings, market_cap, PB ratio, PE ratio, or PS ratio).
    """
    url = f"{BASE_URL}/v1/companies/top/?classifications={dimension}&n_stock=3"
    return retrieve_from_endpoint(url)


@function_tool
def find_companies_screener(order_by: str, limit: int, where: str = "") -> str:
    """
    Screen and rank IDX companies via the Sectors v2 API.

    IMPORTANT:
      - `order_by` and `limit` are always required.
      - `where` is optional — only provide it when a genuine filter condition exists.
        For pure ranking queries (YTD gainers, largest companies, etc.) pass where="" or omit it.

    FIELD REFERENCE:

      Performance (no brackets):
        yearly_mcap_change   Trailing 1-year market-cap % change (decimal: 0.25 = +25%)
        daily_close_change   1-day price change (decimal)
        last_close_price     Latest closing price (IDR)
        ytd_high_price       YTD highest price
        ytd_low_price        YTD lowest price

      Size (no brackets):
        market_cap           Total market cap (IDR)
        market_cap_rank      Cap rank (1=largest; use < for large-cap filter)

      Valuation (no brackets):
        pe_ttm, pb_mrq, ps_ttm, forward_pe

      Profitability (no brackets):
        roe_ttm, roa_ttm

      Dividends:
        yield_ttm            Dividend yield TTM (decimal)
        total_yield[YYYY]    Annual dividend yield (decimal)
        dividend_ttm         Dividend per share TTM

      Financials — MUST use bracket notation:
        revenue[YYYY], earnings[YYYY], eps[YYYY], ebitda[YYYY]
        roe[YYYY], eps_growth[YYYY]

      Forecasts — MUST use bracket notation:
        forecast_eps_growth[YYYY], forecast_revenue_growth[YYYY]

      Category filters (where only):
        sector, sub_sector, industry, listing_date
        indices in ['LQ45','IDX30','KOMPAS100']
        esg_score

    EXAMPLES:
      Top 10 by market cap:           order_by='-market_cap', limit=10
      Top 10 banks by market cap:     where="sub_sector = 'banks'", order_by='-market_cap', limit=10
      Div yield >8% both years:       where='total_yield[2023] > 0.08 and total_yield[2024] > 0.08', order_by='-total_yield[2024]', limit=10
      Low PE (exclude negatives):     where='pe_ttm > 0 and pe_ttm < 12', order_by='pe_ttm', limit=10
      High ROE 2024:                  where='roe[2024] > 0.15', order_by='-roe[2024]', limit=10
      LQ45 index members:             where="indices in ['LQ45']", order_by='-market_cap', limit=45
      Top revenue 2024:               order_by='-revenue[2024]', limit=10
      Large-cap YTD gainers:          where='market_cap_rank < 200', order_by='-yearly_mcap_change', limit=10

    NOTE: Do NOT use this tool for YTD capital gain / YTD performance queries on LQ45/IDX30/KOMPAS100.
          Use get_ytd_capital_gain instead, which computes true YTD return from 2026-01-01.

    :param order_by: Sort field. Prefix with - for descending. Required.
    :param limit: Number of results (max 200). Required.
    :param where: SQL-like filter. Optional — omit for pure ranking queries.
    """
    params = [
        f"order_by={requests.utils.quote(order_by)}",
        f"limit={limit}",
    ]
    if where and where.strip():
        params.insert(0, f"where={requests.utils.quote(where.strip())}")
    url = f"{BASE_URL}/v2/companies/?" + "&".join(params)
    return retrieve_from_endpoint(url)


@function_tool
def get_ytd_capital_gain(index: str = "LQ45", top_n: int = 10) -> str:
    """
    Compute true YTD capital gain (%) for all members of a given index,
    filtered from 2026-01-01 to today, and return the top N ranked by YTD gain descending.

    This is the CORRECT tool to use for queries about:
      - YTD best/worst performers
      - Capital gain % since start of year
      - YTD performance ranking within LQ45, IDX30, or KOMPAS100

    It fetches historical daily prices for each member from 2026-01-01 to today,
    computes YTD return as (last_close - first_close) / first_close * 100,
    and returns the top_n tickers sorted by YTD gain descending.

    :param index: Index name — one of 'LQ45', 'IDX30', 'KOMPAS100'. Default: 'LQ45'
    :param top_n: Number of top performers to return. Default: 10
    """
    import time
    from datetime import date

    today = date.today().isoformat()
    start = "2026-01-01"

    # Step 1: fetch all members of the index
    where = requests.utils.quote(f"indices in ['{index}']")
    members_url = f"{BASE_URL}/v2/companies/?where={where}&order_by=-market_cap&limit=100"
    try:
        resp = requests.get(members_url, headers=headers)
        resp.raise_for_status()
        members_data = resp.json()
    except Exception as e:
        return json.dumps({"error": str(e)})

    results_raw = members_data if isinstance(members_data, list) else members_data.get("results", [])
    tickers = [r["symbol"] for r in results_raw if r.get("symbol")]
    print(f"    {len(tickers)} {index} members found")

    if not tickers:
        return json.dumps({"error": f"No tickers found for index {index}"})

    # Step 2: fetch YTD historical prices for each ticker and compute capital gain
    # Endpoint: GET /v1/daily/{ticker}/?start=YYYY-MM-DD&end=YYYY-MM-DD
    gains = []
    failed = []
    for i, ticker in enumerate(tickers):
        if i > 0:
            time.sleep(0.35)  # avoid 429 rate-limit errors (~3 req/s)

        url = f"{BASE_URL}/v1/daily/{ticker}/?start={start}&end={today}"
        try:
            r = requests.get(url, headers=headers)
            r.raise_for_status()
            raw = r.json()
        except requests.exceptions.HTTPError as e:
            status = e.response.status_code if e.response is not None else "?"
            failed.append(ticker)
            continue
        except Exception as e:
            failed.append(ticker)
            continue

        # Response may be a list directly, or a dict with a data/results key
        if isinstance(raw, list):
            prices = raw
        elif isinstance(raw, dict):
            prices = raw.get("data") or raw.get("results") or raw.get("prices") or []
        else:
            prices = []

        if len(prices) < 2:
            continue

        prices_sorted = sorted(prices, key=lambda x: x.get("date", ""))
        first_close = prices_sorted[0].get("close")
        last_close  = prices_sorted[-1].get("close")

        if first_close and last_close and first_close != 0:
            ytd_gain = (last_close - first_close) / first_close * 100
            gains.append({
                "ticker": ticker,
                "ytd_capital_gain_pct": round(ytd_gain, 2),
                "last_close_price": last_close,
                "ytd_start_price": first_close,
            })

    if failed:
        print(f"    Skipped {len(failed)} ticker(s) due to fetch errors")

    gains.sort(key=lambda x: x["ytd_capital_gain_pct"], reverse=True)
    top = gains[:top_n]

    print(f"    YTD gains computed for {len(gains)} tickers; returning top {top_n}")
    return json.dumps({
        "index": index,
        "period": f"{start} to {today}",
        "top_n": top_n,
        "results": top,
    })

print("Tools defined.")

### Step 2: Visualization
COLORS = ["#2d6a9f", "#f0b429", "#5b9bd5", "#e05c5c", "#6abf69"]

_PRICE_FIELDS = {"last_close_price", "ytd_start_price", "market_cap", "revenue", "earnings"}
_PCT_FIELDS   = {"ytd_capital_gain_pct", "yearly_mcap_change", "yield_ttm", "roe_ttm",
                 "roa_ttm", "daily_close_change", "pe_ttm", "pb_mrq"}

def _is_price_scale(field: str) -> bool:
    """Return True if the field is on a large absolute scale (IDR price/cap), False if % or ratio."""
    f = field.lower()
    if f in _PRICE_FIELDS:
        return True
    if f in _PCT_FIELDS:
        return False
    if "price" in f or "cap" in f or "revenue" in f or "earning" in f:
        return True
    return False


def _plot_single_metric(ax, df, metric, color, label_col="label"):
    values  = df[metric].fillna(0).tolist()
    max_abs = max((abs(v) for v in values if v), default=1)
    n_r     = len(df)
    bars    = ax.barh(
        range(n_r), values,
        height=0.55,
        color=color,
        alpha=0.85,
    )
    for bar, val in zip(bars, values):
        if val and pd.notna(val):
            ax.text(
                bar.get_width() + max_abs * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{val:,.2f}",
                va="center", ha="left", fontsize=7.5, color="#333",
            )
    ax.set_yticks(list(range(n_r)))
    ax.set_yticklabels(df[label_col].tolist(), fontsize=8.5)
    ax.set_xlabel(metric.replace("_", " ").title(), fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_facecolor("#f9fbfd")
    ax.xaxis.grid(True, linestyle=":", alpha=0.4, color="#ccc")
    ax.set_axisbelow(True)


def render_visualizations(payload: dict, query: str) -> None:
    metric_fields = payload.get("metric_fields", [])
    results       = payload.get("results", [])

    if not results:
        print("No results to visualize.")
        return

    df = pd.DataFrame(results)
    for col in metric_fields:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    valid_metrics = [
        c for c in metric_fields
        if c in df.columns and df[c].notna().any()
    ]

    if valid_metrics:
        df = df.sort_values(valid_metrics[0], ascending=False).reset_index(drop=True)

    df["label"] = (
        df["ticker"] + "  "
        + df.get("company_name", pd.Series([""] * len(df))).str[:20]
    )

    n_r = len(df)

    # Separate metrics into groups by scale to avoid combining incompatible units
    pct_metrics   = [m for m in valid_metrics if not _is_price_scale(m)]
    price_metrics = [m for m in valid_metrics if _is_price_scale(m)]

    subplot_groups = []
    if pct_metrics:
        subplot_groups.append(pct_metrics)
    if price_metrics:
        subplot_groups.append(price_metrics)
    if not subplot_groups:
        subplot_groups = [valid_metrics]

    n_plots = len(subplot_groups)
    fig_h   = max(5, n_r * 0.55 + 1.5)
    title   = query if len(query) <= 72 else query[:69] + "..."

    if n_plots == 1:
        fig, axes = plt.subplots(1, 1, figsize=(10, fig_h))
        axes = [axes]
    else:
        fig, axes = plt.subplots(1, n_plots, figsize=(10 * n_plots, fig_h))

    fig.suptitle(title, fontsize=11, fontweight="bold", y=1.01)
    fig.patch.set_facecolor("#ffffff")

    color_offset = 0
    for ax, group in zip(axes, subplot_groups):
        if len(group) == 1:
            _plot_single_metric(ax, df, group[0], COLORS[color_offset % len(COLORS)])
        else:
            bar_h = 0.6 / len(group)
            for i, metric in enumerate(group):
                offset  = (i - (len(group) - 1) / 2) * bar_h
                values  = df[metric].fillna(0).tolist()
                max_abs = max((abs(v) for v in values if v), default=1)
                bars    = ax.barh(
                    [p + offset for p in range(n_r)], values,
                    height=bar_h,
                    label=metric.replace("_", " ").title(),
                    color=COLORS[(color_offset + i) % len(COLORS)],
                    alpha=0.85,
                )
                for bar, val in zip(bars, values):
                    if val and pd.notna(val):
                        ax.text(
                            bar.get_width() + max_abs * 0.01,
                            bar.get_y() + bar.get_height() / 2,
                            f"{val:,.2f}",
                            va="center", ha="left", fontsize=7.5, color="#333",
                        )
            ax.set_yticks(list(range(n_r)))
            ax.set_yticklabels(df["label"].tolist(), fontsize=8.5)
            ax.set_xlabel(
                " / ".join(m.replace("_", " ").title() for m in group), fontsize=9
            )
            ax.legend(fontsize=8, loc="lower right")
            ax.spines[["top", "right"]].set_visible(False)
            ax.set_facecolor("#f9fbfd")
            ax.xaxis.grid(True, linestyle=":", alpha=0.4, color="#ccc")
            ax.set_axisbelow(True)

        color_offset += len(group)

    plt.tight_layout()
    plt.show()

    keep = ["ticker", "company_name", "sector"] + [
        c for c in valid_metrics if c in df.columns
    ] + [
        c for c in df.columns
        if c not in ["ticker", "company_name", "sector", "label"] + valid_metrics
    ]
    keep = [c for c in keep if c in df.columns and c != "label"]

    t = df[keep].copy()
    t.index = range(1, len(t) + 1)
    t.index.name = "Rank"
    for col in valid_metrics:
        if col in t.columns:
            t[col] = t[col].apply(lambda x: f"{x:,.2f}" if pd.notna(x) else "N/A")
    t.columns = [c.replace("_", " ").title() for c in t.columns]

    print(f"\nSource: Sectors Financial API (sectors.app)")
    print(f"Query : {query}\n")
    print(t.to_string())

print("Visualization defined.")

### Agents Architecture Summary
#	Agent	Pattern	Model	Role
#1	screener_agent	Skills and Tools	gpt-5-mini	Maps any IDX query to order_by/limit and optionally where; omits where entirely for pure ranking queries
#2	researcher_agent	Sequential Chain	gpt-5	Fetches overview + financials per ticker; converts decimal fields to percentages; selects 2-4 relevant metrics
#3	evaluator_agent	Judge and Critic	gpt-5-mini	Grades ticker, company name, and non-null metrics; researcher revises up to 3 times
#Note: Most queries are capped at 10.

# ── Agent 1: Screener ──────────────────────────────────────────────────────
screener_agent = Agent(
    name="IDX Screener",
    instructions=(
        "You screen Indonesia Stock Exchange (IDX) companies by calling the appropriate tool.\n\n"
        "CRITICAL RULE — YTD / capital gain queries on an index (LQ45, IDX30, KOMPAS100):\n"
        "  ALWAYS call get_ytd_capital_gain.\n"
        "  When `get_ytd_capital_gain` is called, return the raw JSON output from the tool, not just the tickers. For all other queries, return ONLY a Python list of clean ticker symbols without .JK suffix.\n"
        "  'yearly_mcap_change' is a trailing 1-year metric and must NOT be used for YTD queries.\n"
        "  YTD means from 2026-01-01 to today.\n\n"
        "QUERY -> TOOL MAPPING:\n"
        "  YTD best performers / capital gain % in LQ45/IDX30/KOMPAS100:\n"
        "      -> call get_ytd_capital_gain(index='LQ45', top_n=10)\n"
        "  Largest by market cap:                 find_companies_screener(order_by='-market_cap', limit=10)\n"
        "  Top banks by market cap:               find_companies_screener(where=\"sub_sector = 'banks'\", order_by='-market_cap', limit=10)\n"
        "  Tech sector:                           find_companies_screener(where=\"sector = 'Technology'\", order_by='-market_cap', limit=10)\n"
        "  Consumer goods:                        find_companies_screener(where=\"sector = 'Consumer Non-Cyclicals'\", order_by='-revenue[2024]', limit=10)\n"
        "  Dividend yield >8% both years:         find_companies_screener(where='total_yield[2023] > 0.08 and total_yield[2024] > 0.08', order_by='-total_yield[2024]', limit=10)\n"
        "  Highest dividend yield:                find_companies_screener(order_by='-yield_ttm', limit=10)\n"
        "  Low PE (exclude negatives):            find_companies_screener(where='pe_ttm > 0 and pe_ttm < 12', order_by='pe_ttm', limit=10)\n"
        "  Undervalued (low PB):                  find_companies_screener(where='pb_mrq > 0 and pb_mrq < 1', order_by='pb_mrq', limit=10)\n"
        "  High ROE 2024:                         find_companies_screener(where='roe[2024] > 0.15', order_by='-roe[2024]', limit=10)\n"
        "  LQ45 members by market cap:            find_companies_screener(where=\"indices in ['LQ45']\", order_by='-market_cap', limit=45)\n"
        "  Top revenue 2024:                      find_companies_screener(order_by='-revenue[2024]', limit=10)\n"
        "  EPS growth >20%:                       find_companies_screener(where='eps_growth[2024] > 0.2', order_by='-eps_growth[2024]', limit=10)\n"
        "  Forecast EPS growth:                   find_companies_screener(where='forecast_eps_growth[2025] > 0.2', order_by='-forecast_eps_growth[2025]', limit=10)\n"
        "  Top companies by dimension:            get_top_companies_ranked(dimension='market_cap')\n"
        "  Mid/large cap only: add 'and market_cap_rank < 200' to where.\n"
        "  Listed >2 years: add 'and listing_date < \'2023-01-01\'' to where.\n\n"
        "After calling the tool, return ONLY a Python list of clean ticker symbols without .JK suffix.\n"
        "Example: ['BBCA', 'BBRI', 'TLKM', 'ASII', 'BMRI']"
    ),
    tools=[find_companies_screener, get_ytd_capital_gain, get_top_companies_ranked],
    model="gpt-5-mini",
    output_type=str,
)

# ── Agent 2: Researcher ────────────────────────────────────────────────────
researcher_agent = Agent(
    name="IDX Researcher",
    instructions="""
        You are a financial researcher for Indonesian stocks.
        You receive a user query and a list of IDX ticker symbols.
        Always call get_company_overview for every ticker.
        Also call get_company_financials for each ticker if the query mentions
        dividends, PE, ROE, revenue, earnings, or EPS.

        Pick 2-4 numeric fields most relevant to the query:
          YTD performance / capital gain -> ytd_capital_gain_pct (%), last_close_price
          Dividends        -> yield_2023 (%), yield_2024 (%), yield_ttm (%)
          Valuation        -> pe_ttm, pb_mrq, market_cap
          Revenue/earnings -> revenue_2024, earnings_2024
          Market cap       -> market_cap, market_cap_rank
          General          -> market_cap, last_close_price

        IMPORTANT: If the screener already provided ytd_capital_gain_pct values in the tickers context,
        use those values directly — do NOT replace them with yearly_mcap_change from the overview.

        CONVERSION — apply before writing to output:
          yearly_mcap_change: x100  (0.25 -> 25.0, represents trailing 1-year % change)
          ytd_capital_gain_pct: already in %, use as-is
          yield_ttm, total_yield[YYYY]: x100  (0.08 -> 8.0)
          roe_ttm, roa_ttm, roe[YYYY]: x100
          All other fields: use as-is

        Return ONLY a raw JSON object — no markdown fences, no explanation:
        {"metric_fields": ["field1", "field2"], "results": [{"ticker": "X", "company_name": "Y", "sector": "Z", "field1": 12.5, "field2": 4200}]}
    """,
    tools=[get_company_overview, get_company_financials],
    model="gpt-5",
    output_type=str,
)


# ── Agent 3: Evaluator ─────────────────────────────────────────────────────
@dataclass
class EvaluationFeedback:
    feedback: str
    score: Literal["pass", "expect_improvement", "fail"]

evaluator_agent = Agent(
    name="Evaluator",
    instructions=(
        "Grade the JSON object on three criteria:\n"
        "  1. Every result has a non-empty 'ticker'.\n"
        "  2. Every result has a non-empty 'company_name'.\n"
        "  3. Fields in 'metric_fields' exist in results and at least some rows "
        "have non-null numeric values. All-null metrics = failure.\n"
        "Scoring: 3/3 -> pass | 2/3 -> expect_improvement | fewer -> fail."
    ),
    model="gpt-5-mini",
    output_type=EvaluationFeedback,
)

print("Agents defined.")

### Step 4: Orchestration
# INSERT QUERIES HERE
# Accepts any query about the Indonesian stock market, for example:

# "Top 10 IDX banks by market cap"
# "Indonesian tech stocks with ROE above 15% in 2024"
# "Top 5 IDX consumer goods companies by revenue growth"
# "Stocks in the LQ45 index with PE ratio below 10"
# "Top 10 Indonesian stocks with dividend yield above 8% in the past 2 years"

MAX_CRITIQUE_ITERATIONS = 3

async def main():
    # ── Change the query here ─────────────────────────────────────────────
    query = "Top 10 YTD best performing LQ45 ticker (based on capital gain %), sorted in descending order"
    # ──────────────────────────────────────────────────────────────────────

    print(f"Query: {query}")
    print("=" * 65)

    # ── Stage 1: Screen ────────────────────────────────────────────────────
    print("[Agent 1] Screener running...")
    s = await Runner.run(screener_agent, query)

    tickers = []
    ytd_context = ""

    try:
        # Attempt to parse s.final_output as JSON first (for YTD queries)
        parsed_output = json.loads(s.final_output)
        # Check if the parsed output is likely from get_ytd_capital_gain
        if isinstance(parsed_output, dict) and "results" in parsed_output and any("ytd_capital_gain_pct" in r for r in parsed_output["results"] if isinstance(r, dict)):
            ytd_context = s.final_output # Keep the full JSON as ytd_context
            tickers = [r["ticker"] for r in parsed_output["results"] if "ticker" in r]
            print(f"  (Detected YTD JSON output from screener)")
        else:
            # Fallback to string list parsing if not YTD JSON
            tickers = re.findall(r"'([A-Z0-9]{2,10})'", s.final_output)
    except json.JSONDecodeError:
        # If not JSON, assume it's a string representation of a list of tickers
        tickers = re.findall(r"'([A-Z0-9]{2,10})'", s.final_output)

    print(f"  Tickers ({len(tickers)}): {tickers}\n")

    if not tickers:
        print("Screener returned no tickers.")
        print(f"Raw output:\n{s.final_output}")
        return

    # ── Stage 2: Research ──────────────────────────────────────────────────
    print("[Agent 2] Researcher fetching data...")
    researcher_input = f"User query: '{query}'\nTickers: {tickers}"
    if ytd_context:
        researcher_input += (
            f"\n\nPre-computed YTD capital gain data (use ytd_capital_gain_pct values as-is, "
            f"do NOT replace with yearly_mcap_change):\n{ytd_context}"
        )
    r = await Runner.run(researcher_agent, researcher_input)
    current = r.final_output
    print(f"  Research complete ({len(current)} chars)\n")

    # ── Stage 3: Evaluate and revise ──────────────────────────────────────
    final = None
    for i in range(1, MAX_CRITIQUE_ITERATIONS + 1):
        print(f"[Agent 3] Evaluator grading (attempt {i}/{MAX_CRITIQUE_ITERATIONS})...")
        ev = await Runner.run(evaluator_agent, current)
        fb = ev.final_output_as(EvaluationFeedback)
        print(f"  Score: {fb.score.upper()}")
        print(f"  Feedback: {fb.feedback}\n")

        if fb.score == "pass":
            final = current
            break

        r = await Runner.run(
            researcher_agent,
            f"User query: '{query}'\nTickers: {tickers}\n"
            f"Previous output graded '{fb.score}'.\nFeedback: {fb.feedback}\n"
            "Fix the issues and return the corrected JSON object."
        )
        current = r.final_output
        if i == MAX_CRITIQUE_ITERATIONS:
            print("Max iterations reached. Using best available output.\n")
            final = current

    # ── Render ─────────────────────────────────────────────────────────────
    print("=" * 65)
    try:
        clean  = re.sub(r"```[a-z]*\n?", "", final).strip().rstrip("`")
        parsed = json.loads(clean)
        render_visualizations(parsed, query)
    except (json.JSONDecodeError, TypeError) as e:
        print(f"JSON parsing failed: {e}\nRaw output:\n{final}")


if __name__ == "__main__":
    asyncio.run(main())