import asyncio
import os
from dotenv import load_dotenv
from typing import List, Dict, Any
from pydantic import BaseModel
import json
from agents import Agent, Runner, function_tool, trace, AgentOutputSchema
from utils.api_client import retrieve_from_endpoint


class QueryValues(BaseModel):
    """Model for query values in the results"""
    pass  # This will be dynamically populated

class CompanyResult(BaseModel):
    """Model for individual company result"""
    symbol: str
    company_name: str
    query_values: Dict[str, Any]

class ScreenerResults(BaseModel):
    """Model for the complete screener results"""
    results: List[CompanyResult]


@function_tool
def get_company_overview(ticker: str, country: str) -> str:
    """
    Get company overview from Singapore Exchange (SGX) or Indonesia Exchange (IDX)
    """
    assert country.lower() in ["indonesia", "singapore", "malaysia"], "Country must be either Indonesia, Singapore, or Malaysia"

    if(country.lower() == "indonesia"):
        url = f"https://api.sectors.app/v1/company/report/{ticker}/?sections=overview"
    if(country.lower() == "singapore"):
        url = f"https://api.sectors.app/v1/sgx/company/report/{ticker}/"
    if(country.lower() == "malaysia"):
        url = f"https://api.sectors.app/v1/klse/company/report/{ticker}/"

    try:
        return retrieve_from_endpoint(url)
    except Exception as e:
        print(f"Error occurred: {e}")
        return None

@function_tool
def  find_companies_screener(query: str) -> List[str]:
    url =  f"https://api.sectors.app/v2/companies/?q={query}"
    return retrieve_from_endpoint(url)

@function_tool
def write_research_to_file(ticker: str, overview: str, filename: str) -> str:
    """
    Write research note to a file in the output folder. Creates the output folder if it doesn't exist.
    """
    # Create output folder if it doesn't exist
    output_folder = "output"
    os.makedirs(output_folder, exist_ok=True)
    
    # Construct full file path
    full_path = os.path.join(output_folder, filename)
    
    content = f"# Research Note for {ticker}\n\n{overview}\n" 
    with open(full_path, "a") as f:
        f.write(content)
    
    return f"Research note for {ticker} written to {full_path}."

@function_tool
def open_sectors_search_by_ticker(ticker: str) -> str:
    import webbrowser
    base_url = "https://sectors.app"
    search_url = f"{base_url}/search/?query={ticker}"
    company_url = f"{base_url}/idx/{ticker}/"
    webbrowser.open_new_tab(search_url)
    webbrowser.open_new_tab(company_url)
    return f"Opened research pages for {ticker} in web browser."


browser_agent = Agent(
    name="browser_agent",
    instructions="Open the sectors.app search page and company page for the given ticker in a web browser.",
    tools=[open_sectors_search_by_ticker],
    model="gpt-5-nano", # most cost-effective and lowest latency model output_type=str
)

natural_language_screener = Agent(
    name="natural_language_screener",
    instructions="""Get list of companies based on a natural language query and return structured data in the specified JSON format.
    You must return data in this exact format:
    {
        "results": [
            {
                "symbol": "AMAR.JK",
                "company_name": "PT Bank Amar Indonesia Tbk.",
                "query_values": {
                    "non_performing_loan[2024]": 297477000000,
                    "gross_loan[2024]": 2929997000000,
                    "(non_performing_loan[2024]/gross_loan[2024])": 0.101528090301799
                }
            },
        ]
    }
    
    Use the find_companies_screener tool to get company data, then structure it according to the format above.
    The query_values should contain the actual financial metrics and calculations relevant to the user's query.
    """,
    tools=[find_companies_screener],
    model="gpt-5-mini", # balanced cost & performance, great for structured data extraction or Tool use
    output_type=AgentOutputSchema(ScreenerResults, strict_json_schema=False),
)

research_writer_agent = Agent(
    name="research_writer_agent",
    instructions="""You are a research writer agent. Given a query and a list of tickers, you will:
    1. Write the initial header with the query
    2. Research each company ticker using get_company_overview (assume Indonesian companies)
    3. Write each research result to a file inside the "output" folder
    4. Use the ticker as the filename with .txt extension (e.g., "BBRI.JK.txt")
    
    For each ticker, get the company overview and then write it to file using write_research_to_file.
    The filename should be the ticker symbol followed by .txt extension.
    Handle failures gracefully.""",
    tools=[get_company_overview, write_research_to_file],
    model="gpt-5",
    output_type=str
)

async def main():
    input_prompt = input(f"🤖: I'm a universal stock screener by Sectors. Tell what is on your brain ? \n👧: ")
    # input_prompt = "top 3 companies based on npl ratio"

    # Ensure the entire workflow is a single trace
    with trace("Sequential research flow"):
        # 1. The NLP Screener Agent interprets the user's query
        screened_companies_result = await Runner.run(
            natural_language_screener,
            input_prompt
        )
        
        # Convert the result to a dictionary and format as JSON
        result_dict = screened_companies_result.final_output.model_dump()
        formatted_json = json.dumps(result_dict, indent=2, ensure_ascii=False)
        
        print("🤖: Researching ...")
        print(formatted_json)
        
        # Extract tickers from the structured results
        screener_data = screened_companies_result.final_output
        tickers_list = [company.symbol for company in screener_data.results]
        print("🤖: Found companies:", tickers_list)

        # 2. The Research Writer Agent research each company and write the report in a file inside a folder name "Output".
        for ticker in tickers_list:
            await Runner.run(
                research_writer_agent,
                f"Research {ticker}"
            )

        print(f"Done! I have provided the information on: {input_prompt} and written the report to the Output folder.")

        # 3. Open the sectors.app pages for the last company's ticker
        open_sectors_app = input("Do you want to open the sectors.app page for the last company on this list too? (y/n): ").lower()
        if open_sectors_app == 'y':
            last_ticker = tickers_list[-1]  
            await Runner.run(
                browser_agent,
                f"Open sectors.app page for {last_ticker}"
            )
            print(f"I have opened the sectors.app search and company pages for the ticker {last_ticker} in your web browser. You can now review financial data, news, and analysis related to {last_ticker} directly on those pages. If you need a summary or specific information from those pages, let me know!")
        else:
            print("Done!")

if __name__ == "__main__":
    asyncio.run(main())