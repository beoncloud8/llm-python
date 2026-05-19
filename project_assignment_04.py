import os
import asyncio
import json
import requests
from typing import List
from agents import Agent, Runner, function_tool
from dotenv import load_dotenv

load_dotenv()

SECTORS_API_KEY = os.getenv("SECTORS_API_KEY")
if not SECTORS_API_KEY:
    raise ValueError("SECTORS_API_KEY environment variable is not set")
headers = {"Authorization": SECTORS_API_KEY}

def retrieve_from_endpoint(url: str) -> str:
    try:
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        data = response.json()
    except requests.exceptions.HTTPError as err:
        raise RuntimeError(f"HTTP error occurred: {err}") from err
    except requests.exceptions.RequestException as err:
        raise RuntimeError(f"Request error occurred: {err}") from err
    except json.JSONDecodeError as err:
        raise RuntimeError(f"JSON decode error: {err}") from err
    return json.dumps(data)

@function_tool
def get_company_overview(ticker: str, country: str) -> str:
    """
    Get company overview from Singapore Exchange (SGX), Bursa Malaysia (KLSE), or Indonesia Exchange (IDX)
    """
    valid_countries = ["indonesia", "singapore", "malaysia"]
    country_lower = country.lower()
    
    if country_lower not in valid_countries:
        raise ValueError(f"Country must be one of {valid_countries}")
    
    if not ticker or not ticker.strip():
        raise ValueError("Ticker cannot be empty")

    # Returns a comprehensive company report organized into distinct sections. By default all sections are included. 
    # Use sections to request only the data you need and reduce response size.
    if country.lower() == "indonesia":
        url = f"https://api.sectors.app/v2/company/report/{ticker}/?sections=overview"
    elif country.lower() == "singapore":
        url = f"https://api.sectors.app/v2/sgx/company/report/{ticker}/"
    elif country.lower() == "malaysia":
        url = f"https://api.sectors.app/v2/klse/company/report/{ticker}/"

    try:
        return retrieve_from_endpoint(url)
    except Exception as e:
        error_msg = f"Error retrieving company overview for {ticker}: {e}"
        print(error_msg)
        return json.dumps({"error": error_msg})

@function_tool
def find_companies_screener(query: str) -> str:
    """
    High-performance API for filtering and sorting IDX-listed companies. 
    Supports both structured SQL-like queries (where, order_by) and natural language queries (q). 
    """
    
    if not query or not query.strip():
        raise ValueError("Query cannot be empty")
    
    url = f"https://api.sectors.app/v2/companies/?q={query.strip()}"
    
    try:
        return retrieve_from_endpoint(url)
    except Exception as e:
        error_msg = f"Error searching companies: {e}"
        print(error_msg)
        return json.dumps({"error": error_msg})

research_assistant = Agent(
    name="Research Assistant",
    instructions="""Your are a financial research assistant that cover stock exchange information from Indonesia, Singapore, and Malaysia.
    If the user query ask report of information by providing company's {ticker}, use get_company_overview tool.
    If the user query is not clear what companies, ask for company's ticker.
    If the user query ask for general information of Indonesian companies or based on certain criteria, use find_companies_screener tool.
    """,
    tools=[
        get_company_overview, 
        find_companies_screener,
    ],
    tool_use_behavior="run_llm_again"
)

async def main():
    query = input("Enter your query: ")
    result = await Runner.run(
        research_assistant,
        query
    )
    print(f"😊: {query}")
    print(f"🤖: {result.final_output}")
    
if __name__ == "__main__":
    asyncio.run(main())