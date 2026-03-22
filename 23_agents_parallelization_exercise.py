"""
This example shows the parallelization pattern. We run the agent three times in parallel, and pick
the best result.

# Usage:
🤖: I'm a financial report research analyst. Enter a stock ticker on IDX to begin. 
👧: ADRO
"""

import asyncio

from agents import Agent, Runner, ItemHelpers, function_tool, trace
from typing import List
from utils.api_client import retrieve_from_endpoint

@function_tool
def find_companies_screener(query: str) -> List[str]:
    url =  f"https://api.sectors.app/v2/companies/?q={query}"
    return retrieve_from_endpoint(url)

@function_tool
def get_company_financials(ticker: str) -> str:
    """
    Get company financials from Indonesia Exchange (IDX)
    """
    url = f"https://api.sectors.app/v1/company/report/{ticker}/?sections=financials"
    try:
        return retrieve_from_endpoint(url)
    except Exception as e:
        print(f"Error occurred: {e}")
        return None

       
@function_tool
def get_revenue_segments(ticker: str) -> str:
    """
    Get revenue segments for a company from Indonesia Exchange (IDX)
    """
    
    url = f"https://api.sectors.app/v1/company/get-segments/{ticker}/"
    try:
        return retrieve_from_endpoint(url)
    except Exception as e:
        print(f"Error occurred: {e}")
        return None


@function_tool
def get_quarterly_financials(ticker: str) -> str:
    """
    Get revenue segments for a company from Indonesia Exchange (IDX)
    """
    
    url = f"https://api.sectors.app/v1/financials/quarterly/{ticker}/?report_date=2024-12-31&approx=true"
    try:
        return retrieve_from_endpoint(url)
    except Exception as e:
        print(f"Error occurred: {e}")
        return None


company_financials_research_agent = Agent(
    name="company_financials_research_agent",
    instructions="Research the financials of a company based on the ticker provided.",
    tools=[get_company_financials],
    output_type=str
)

company_revenue_breakdown_agent = Agent(
    name="company_revenue_breakdown_agent",
    instructions="Research the revenue breakdown of a company based on the ticker provided.",
    tools=[get_revenue_segments],
    output_type=str
)

company_quarterly_financials_agent = Agent(
    name="company_quarterly_financials_agent",
    instructions="Research the quarterly financials of a company based on the ticker provided.",
    tools=[get_quarterly_financials],
    output_type=str
)

research_team_leader_aggregator = Agent(
    name="research_team_leader_aggregator",
    instructions="""You are the team leader of a research team. You will aggregate the results from these agents and provide a consolidated analysis in the following EXACT markdown structure:

**1. Growth and Profitability**
**Earnings and Income**
- **Net Income growth**: [Analyze net income trends across years]
- **EPS growth**: [Discuss EPS performance and projections]
- **Revenue**: [Analyze revenue growth and sources]
- [Add 2-3 more key income metrics]

**Profitability Ratios**
- **Net Profit Margins**: [Discuss margin trends]
- **ROA and ROE**: [Analyze return metrics]
- **Efficiency**: [Discuss operational efficiency ratios]
- [Add 1-2 more profitability metrics]

---
**2. Balance Sheet Strength & Asset Growth**
- **Total assets**: [Discuss asset growth trends]
- **Loan Book**: [Analyze lending portfolio]
- **Deposits**: [Discuss funding structure]
- **Liquidity Ratios**: [Analyze liquidity position]
- [Add 1-2 more balance sheet metrics]

---
**3. Revenue and Expense Structure**
**Revenue**
- **Primary Revenue Sources**: [Break down main revenue streams]
- **Secondary Revenue**: [Discuss other income sources]
- **Total Revenue**: [Summarize revenue performance]

**Expenses**
- **Operating Expenses**: [Analyze cost structure]
- **Provision for Losses**: [Discuss risk management]
- [Add other key expense categories]

---
**4. Quarterly Snapshot**
- **Quarterly Revenue**: [Latest quarterly performance]
- **Quarterly Profit**: [Quarterly profitability]
- **Balance Sheet Health**: [Latest balance sheet metrics]
- **Liquidity Position**: [Quarterly liquidity metrics]

---
**5. Key Takeaways and Strategic Position**
- **Industry Position**: [Company's market standing]
- **Growth Outlook**: [Future growth prospects]
- **Risk Factors**: [Key risks and challenges]
- **Strategic Advantages**: [Competitive strengths]

---
**Summary Table – Key Metrics**
| Metric | 2023 Actual | 2024 (F) | 2025 (F) |
|--------|-------------|-----------|-----------|
| EPS (IDR) | [value] | [value] | [value] |
| Revenue (IDR T) | [value] | [value] | [value] |
| Net Income (IDR T) | [value] | [value] | [value] |
| ROA (%) | [value] | [value] | [value] |
| ROE (%) | [value] | [value] | [value] |
| Net Profit Margin (%) | [value] | [value] | [value] |
| [Add 2-3 more key metrics] | [value] | [value] | [value] |

---
**If you need:**
- Detailed segment/industry/sector breakdown
- Historical multi-year trend & peer comparison
- Specific ratio analysis (efficiency, cost/income, NPL, etc.)
- Insights on dividend policy or a forecast scenario
**Just let me know!**

IMPORTANT: Use the provided financial data to fill in the [bracketed] sections with actual analysis and values. Be comprehensive and data-driven.""",
    output_type=str
)

natural_language_screener = Agent(
    name="natural_language_screener",
    instructions="Get the company name and basic information based on the ticker provided.",
    tools=[find_companies_screener],
    output_type=str
)

async def parallel_stock_research(ticker: str) ->  str: 
    # Run the three research agents in parallel 
    financials_task = asyncio.create_task(
        Runner.run(company_financials_research_agent, ticker) 
    )
    revenue_task = asyncio.create_task(
        Runner.run(company_revenue_breakdown_agent, ticker)
    )
    quarterly_task = asyncio.create_task(
        Runner.run(company_quarterly_financials_agent, ticker)
    )
    
    financials_result = await financials_task
    revenue_result = await revenue_task
    quarterly_result = await quarterly_task

    # Extract the actual outputs from the Runner results
    financials_output = ItemHelpers.text_message_outputs(financials_result.new_items)
    revenue_output = ItemHelpers.text_message_outputs(revenue_result.new_items)
    quarterly_output = ItemHelpers.text_message_outputs(quarterly_result.new_items)

    # Aggregate results
    aggregated_input = f"""
    Company Financials:
    {financials_output}

    Revenue Breakdown:
    {revenue_output}
    
    Quarterly Financials:
    {quarterly_output}
    """

    # Run the aggregator agent
    consolidated_report = await Runner.run(research_team_leader_aggregator, aggregated_input)
    return consolidated_report.final_output

async def main():
    input_prompt = input(f"🤖: I'm a financial report research analyst. Enter a stock ticker on IDX to begin. \n👧: ")

    # Get company name dynamically
    company_info_result = await Runner.run(natural_language_screener, input_prompt)
    # Extract company name - look for the actual company name in the response
    full_response = company_info_result.final_output
    
    # Try to extract the company name from the response
    # The response might contain the company name in different formats
    if "PT " in full_response:
        # Extract the first occurrence of "PT " until the next period or comma
        import re
        match = re.search(r'PT [^.]+', full_response)
        if match:
            company_name = match.group().strip()
        else:
            company_name = full_response.split('\n')[0].strip()
    else:
        company_name = full_response.split('\n')[0].strip()

    print(f"Here's a consolidated analysis of {company_name} ({input_prompt}) based on your latest provided financials and breakdowns:")
    result = await parallel_stock_research(input_prompt)
    print(result)

if __name__ == "__main__":
    asyncio.run(main())


"""
agent_res1, agent_res2, agent_res3 = await asyncio.gather(
    Runner.run(company_financials_research_agent, input_prompt),
    Runner.run(company_revenue_breakdown_agent, input_prompt),
    Runner.run(company_quarterly_financials_agent, input_prompt)
)
"""
