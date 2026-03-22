#A weekly get_ticker_price agent, and show the result in a simple table format.

import os
from dotenv import load_dotenv

import requests
import json
from agents import Agent, Runner, function_tool

load_dotenv()
OPENAI_DEFAULT_MODEL = os.getenv("OPENAI_DEFAULT_MODEL", "gpt-4")  # Default to gpt-4 if not set

SECTORS_API_KEY= os.getenv("SECTORS_API_KEY")
if not SECTORS_API_KEY:
    raise ValueError("SECTORS_API_KEY environment variable is required")
headers = {"Authorization": SECTORS_API_KEY}

def validate_ticker(ticker: str, country: str) -> bool:
    """Validate ticker format based on exchange requirements"""
    if not ticker or not isinstance(ticker, str):
        return False
    
    country = country.lower()
    
    if country == "malaysia":
        # KLSE tickers are typically 4-digit numeric codes
        return len(ticker) == 4 and ticker.isdigit()
    elif country == "singapore":
        # SGX tickers are alphanumeric, typically 3 or 4 characters
        return len(ticker) <= 4 and ticker.isalnum()
    elif country == "indonesia":
        # IDX tickers are typically 4-letter codes
        return len(ticker) == 4 and ticker.isalpha()
    
    return False

def retrieve_from_endpoint(url: str) -> dict:
	try:
		response = requests.get(url, headers=headers)
		response.raise_for_status()
		data = response.json()
	except requests.exceptions.HTTPError as err:
		raise SystemExit(err)
	return json.dumps(data)

@function_tool
def get_ticker_price(ticker: str, country: str) -> str:
	"""
	Get company overview from Singapore Exchange (SGX) or Indonesia Exchange (IDX)
	"""
	# Validate ticker format for the specific exchange
	if not validate_ticker(ticker, country):
		raise ValueError(f"Invalid ticker format for {country}: {ticker}")

	assert country.lower() in ["indonesia", "singapore", "malaysia"], "Country must be either Indonesia, Singapore, or Malaysia"

	if country.lower() == "indonesia":
		url = f"https://api.sectors.app/v1/company/report/{ticker}/?sections=overview"
	elif country.lower() == "singapore":
		url = f"https://api.sectors.app/v1/sgx/company/report/{ticker}/"
	elif country.lower() == "malaysia":
		url = f"https://api.sectors.app/v1/klse/company/report/{ticker}/"

	try:
		return retrieve_from_endpoint(url)
	except Exception as e:
		print(f"Error occurred for {ticker}: {e}")
		return None

ticker_explore_agent = Agent(
	name="ticker_explore_agent",
	model=OPENAI_DEFAULT_MODEL,
	instructions="""Analyze the user's query and respond accordingly:
 
	If the query asks for "today" or current stock price (like "Show me company's stocks price"), provide only:
	- Current stock price
	- Price change percentage
 
	If the query asks for "this week" or weekly stock price, display results in this exact format:
	[D05]
	| date | stock price | price change percentage |
 
	Only provide the requested information, no extra text.""",
	tools=[get_ticker_price],
	tool_use_behavior="run_llm_again"
)

async def main():
	# Test with today's price query
	query = "Show me company's stocks price that listed on Singapore Exchange with ticker 'D05'."
	result = await Runner.run(
		ticker_explore_agent,
		query
	)
	print(f"👨: {query}")
	print(f"🤖: {result.final_output}")

	print("\n" + "="*50 + "\n")
 
	# Test with weekly price query
	weekly_query = "Tell me this week company's stocks price that listed on Singapore Exchange with ticker 'D05'."
	weekly_result = await Runner.run(
		ticker_explore_agent,
		weekly_query
	)
	print(f"👨: {weekly_query}")
	print(f"🤖: {weekly_result.final_output}")

if __name__ == "__main__":
	import asyncio
	asyncio.run(main())