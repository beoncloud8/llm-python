import os
import json
import requests
from typing import List
from agents import Agent, Runner, function_tool
from dotenv import load_dotenv

load_dotenv()

SECTORS_API_KEY = os.getenv("SECTORS_API_KEY")
headers = {"Authorization": SECTORS_API_KEY}

def retrieve_from_endpoint(url: str) -> dict:
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        data = response.json()
    except requests.exceptions.HTTPError as err:
        raise SystemExit(err)
    return json.dumps(data)

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
def  find_companies_screener(query: str) ->  List[str]:
    url =  f"https://api.sectors.app/v2/companies/?q={query}"
    return retrieve_from_endpoint(url)
   

stock_assistant = Agent(
    name="stock_assistant",
    instructions="Either do stock screening or get company overview based on user query.",
    tools=[
        get_company_overview, 
        find_companies_screener,
    ],
    tool_use_behavior="run_llm_again"
)

async def main():
    query = "Screen for IDX companies where Prajogo Pangestu is a major shareholder."
    result = await Runner.run(
        stock_assistant,
        query
    )
    print(f"👧: {query}")
    print(f"🤖: {result.final_output}")
    
if __name__ == "__main__":
    import asyncio
    asyncio.run(main())