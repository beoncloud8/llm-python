"""
Hybrid Financial Analysis System
Combines serial and parallel patterns for comprehensive financial research.

# Usage:
python project_assignment_06.py
🤖: Welcome to Financial Research Analyst Service
Please select the service you want:
1. Single Company Analysis (Detailed report for one company)
2. Comparative Analysis (Compare 2-4 companies)
3. Exit
"""

import asyncio
import os
from dotenv import load_dotenv
from agents import Agent, Runner, ItemHelpers, function_tool, trace
from utils.api_client import retrieve_from_endpoint
from typing import List
from datetime import datetime
import os.path

load_dotenv()

# Function Tools for Financial Analysis
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
    Get quarterly financials for a company from Indonesia Exchange (IDX)
    """
    url = f"https://api.sectors.app/v1/financials/quarterly/{ticker}/?report_date=2024-12-31&approx=true"
    try:
        return retrieve_from_endpoint(url)
    except Exception as e:
        print(f"Error occurred: {e}")
        return None

@function_tool
def write_research_to_file(ticker: str, content: str) -> str:
    """
    Write research content to a file named with ticker symbol
    """
    try:
        # Create reports directory if it doesn't exist
        os.makedirs("reports", exist_ok=True)
        
        # Generate timestamp filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{ticker}_{timestamp}.txt"
        filepath = os.path.join("reports", filename)
        
        # Save content to file
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        
        return f"Report saved to: {filepath}"
        
    except Exception as e:
        return f"Error saving file: {e}"

# Agents
individual_company_research_agent = Agent(
    name="individual_company_research_agent",
    instructions="""You are a comprehensive financial research analyst. Research the given company ticker and provide a detailed report covering:
1. Company overview and business description
2. Financial performance and metrics
3. Revenue breakdown and business segments
4. Quarterly financial trends
5. Investment analysis and recommendations

Provide a thorough, professional analysis that would be valuable for investment decisions.""",
    tools=[get_company_overview, get_company_financials, get_revenue_segments, get_quarterly_financials, write_research_to_file],
    model="gpt-4",
    output_type=str
)

financial_research_agent = Agent(
    name="financial_research_agent",
    instructions="""You are a financial research analyst. Research the given company ticker and provide comprehensive financial analysis including:
1. Company financials and performance metrics
2. Revenue breakdown and business segments
3. Quarterly financial trends

Provide a detailed analysis that can be compared with other companies.""",
    tools=[get_company_financials, get_revenue_segments, get_quarterly_financials],
    output_type=str
)

research_team_leader_aggregator = Agent(
    name="research_team_leader_aggregator",
    instructions="You are the team leader of a research team. You will aggregate the results from these agents and provide a consolidated answer that is relevant to the user.",
    output_type=str
)

# Menu and Service Functions
def display_menu():
    """Display the main menu and get user choice"""
    print("\n" + "="*60)
    print("🤖: Welcome to Financial Research Analyst Service")
    print("Please select the service you want:")
    print("1. Single Company Analysis (Detailed report for one company)")
    print("2. Comparative Analysis (Compare 2-4 companies)")
    print("3. Exit")
    print("="*60)
    return input("Enter your choice (1-3): ").strip()

def validate_ticker(ticker: str) -> bool:
    """Validate ticker format"""
    ticker = ticker.strip().upper()
    return ticker.isalpha() and len(ticker) >= 3 and len(ticker) <= 6

def validate_multiple_tickers(input_str: str, min_count: int, max_count: int) -> tuple[List[str], str]:
    """Validate multiple tickers input"""
    try:
        # Handle both comma-separated and space-separated inputs
        if ',' in input_str:
            tickers = [ticker.strip().upper() for ticker in input_str.split(',')]
        else:
            tickers = [ticker.strip().upper() for ticker in input_str.split()]
        
        tickers = [ticker for ticker in tickers if ticker]  # Remove empty strings
        
        if len(tickers) < min_count or len(tickers) > max_count:
            return [], f"Please enter between {min_count} and {max_count} tickers. You provided {len(tickers)}."
            
        # Validate ticker format
        for ticker in tickers:
            if not validate_ticker(ticker):
                return [], f"'{ticker}' doesn't look like a valid stock ticker. Tickers should be 3-6 letters."
        
        return tickers, ""
        
    except Exception as e:
        return [], f"Error parsing input: {e}"

async def save_comparison_to_file(content: str, tickers: List[str]):
    """Save comparison report to timestamped file in comparisons folder"""
    try:
        # Create output directory if it doesn't exist
        output_dir = "comparisons"
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate timestamp filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        ticker_str = '_vs_'.join(tickers)
        filename = f"comparison_{ticker_str}_{timestamp}.txt"
        filepath = os.path.join(output_dir, filename)
        
        # Save content to file
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"\n📁 Comparison report saved to: {filepath}")
        
    except Exception as e:
        print(f"❌ Error saving comparison report: {e}")

async def single_company_analysis():
    """Handle single company analysis service"""
    print("\n" + "="*50)
    print("📊 Single Company Analysis")
    print("="*50)
    
    ticker_input = input("Enter the stock ticker (e.g., BBCA): ").strip()
    
    if not validate_ticker(ticker_input):
        print(f"❌ '{ticker_input}' doesn't look like a valid stock ticker. Tickers should be 3-6 letters.")
        return
    
    ticker = ticker_input.upper()
    print(f"🔍 Analyzing company: {ticker}...")
    
    with trace("Single_Company_Analysis"):
        try:
            # Generate comprehensive individual report
            result = await Runner.run(
                individual_company_research_agent,
                f"{ticker}, indonesia"
            )
            
            # Format the report
            report_content = f"SINGLE COMPANY ANALYSIS - {ticker}\n{'='*60}\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n{result.final_output}"
            
            print(f"\n🤖: Analysis Complete!\n\n{report_content}")
            
        except Exception as e:
            print(f"❌ Error during analysis: {e}")

async def comparative_analysis():
    """Handle comparative analysis service"""
    print("\n" + "="*50)
    print("📊 Comparative Analysis")
    print("="*50)
    
    ticker_input = input("Enter 2-4 stock tickers (comma or space separated, e.g., BBCA, BBRI, BREN): ").strip()
    
    tickers, error_msg = validate_multiple_tickers(ticker_input, 2, 4)
    
    if error_msg:
        print(f"❌ {error_msg}")
        return
    
    print(f"🔍 Analyzing companies: {', '.join(tickers)}...")
    
    with trace("Comparative_Analysis"):
        try:
            # Run parallel analysis on all tickers
            results = await asyncio.gather(
                *[Runner.run(financial_research_agent, ticker) for ticker in tickers],
                return_exceptions=True
            )
            
            # Check for exceptions
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    print(f"❌ Error analyzing {tickers[i]}: {result}")
                    results[i] = f"Analysis failed for {tickers[i]}: {str(result)}"
            
            # Extract outputs
            outputs = []
            for i, result in enumerate(results):
                if not isinstance(result, Exception):
                    output_text = "\n".join(ItemHelpers.text_message_outputs(result.new_items))
                    outputs.append(f"=== ANALYSIS FOR {tickers[i]} ===\n{output_text}")
            
            # Aggregate the results
            aggregated_result = "\n\n".join(outputs)
            
            # Create comparison analysis
            comparison_prompt = f"""You are a senior financial analyst. Compare these {len(tickers)} companies based on the research provided:
            
{aggregated_result}

Provide a comprehensive comparison covering:
1. Financial performance comparison
2. Revenue structure differences
3. Growth trends and prospects
4. Investment considerations and risks
5. Relative strengths and weaknesses
6. Ranking and recommendation summary

Be specific and provide actionable insights. Include a clear ranking from most recommended to least recommended investment."""
            
            comparison_result = await Runner.run(
                research_team_leader_aggregator,
                comparison_prompt
            )

            # Format the final output
            final_output = f"COMPARATIVE ANALYSIS - {', '.join(tickers)}\n{'='*60}\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n{comparison_result.final_output}"
            
            print(f"\n🤖: {final_output}")
            
            # Save to file
            await save_comparison_to_file(final_output, tickers)
            
        except Exception as e:
            print(f"❌ Error during analysis: {e}")

async def main():
    """Main application loop with menu system"""
    while True:
        choice = display_menu()
        
        if choice == "1":
            await single_company_analysis()
        elif choice == "2":
            await comparative_analysis()
        elif choice == "3":
            print("\n👋: Thank you for using Financial Research Analyst Service!")
            break
        else:
            print("❌: Invalid choice. Please enter 1, 2, or 3.")
        
        # Ask if user wants to continue
        if choice in ["1", "2"]:
            continue_choice = input("\nWould you like to perform another analysis? (y/n): ").strip().lower()
            if continue_choice not in ['y', 'yes']:
                print("\n👋: Thank you for using Financial Research Analyst Service!")
                break

if __name__ == "__main__":
    asyncio.run(main())
