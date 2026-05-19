"""
This code shows analyze three companies, identified by its stock ticker. The agent is run three times in parallel, and deliver the result
comparisation from those three companies .

# Usage:
🤖: I'm a comparative financial research analyst. Enter three stock ticker from IDX's companies to compare. 
😊: ADRO BBCA BREN
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


async def save_comparison_to_file(content: str, tickers: List[str]):
    """Save comparison report to timestamped file in comparison_output folder"""
    try:
        # Create output directory if it doesn't exist
        output_dir = "comparison_output"
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate timestamp filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        ticker_str = '_vs_'.join(tickers)
        filename = f"comparison_{ticker_str}_{timestamp}.txt"
        filepath = os.path.join(output_dir, filename)
        
        # Save content to file
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"\n📁 Report saved to: {filepath}")
        
    except Exception as e:
        print(f"❌ Error saving report: {e}")


async def main():
    input_prompt = input(f"🤖: I'm a comparative financial research analyst. Enter 2-4 stock tickers from IDX's companies to compare (comma or space separated): \n😊: ")
    
    # Parse and validate input
    try:
        # Handle both comma-separated and space-separated inputs
        if ',' in input_prompt:
            tickers = [ticker.strip().upper() for ticker in input_prompt.split(',')]
        else:
            tickers = [ticker.strip().upper() for ticker in input_prompt.split()]
        
        tickers = [ticker for ticker in tickers if ticker]  # Remove empty strings
        
        if len(tickers) < 2 or len(tickers) > 4:
            print(f"❌ Error: Please enter between 2 and 4 tickers. You provided {len(tickers)}.")
            return
            
        # Validate ticker format (basic validation)
        for ticker in tickers:
            if not ticker.isalpha() or len(ticker) < 3 or len(ticker) > 6:
                print(f"❌ Error: '{ticker}' doesn't look like a valid stock ticker. Tickers should be 3-6 letters.")
                return
                
    except Exception as e:
        print(f"❌ Error parsing input: {e}")
        return
    
    print(f"🔍 Analyzing companies: {', '.join(tickers)}...")
    
    # Ensure the entire workflow is a single trace
    with trace("Parallelization"):
        try:
            # Run the same agent for all tickers in parallel
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
    
if __name__ == "__main__":
    asyncio.run(main())