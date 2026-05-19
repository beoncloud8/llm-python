"""
Integrated Financial Analysis System - Streamlined Version
Combines menu-driven services with inline LinkedIn article generation.

# Usage:
python project_assignment_09.py
"""

import asyncio
import os
import glob
from datetime import datetime
from typing import List, Dict, Any
from pydantic import BaseModel
from agents import Agent, Runner, ItemHelpers, function_tool, trace
from utils.api_client import retrieve_from_endpoint
from dotenv import load_dotenv

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

# Data Models for LinkedIn Article Generation
class ComparisonInsights(BaseModel):
    """Model for comparison insights extracted from reports"""
    companies: List[str]
    key_findings: List[str]
    rankings: List[str]
    recommendations: str
    generated_date: str

# Report Discovery and Parsing Functions
def get_latest_comparison_report() -> str:
    """Find the latest comparison report in comparisons folder"""
    try:
        # Look for comparison files in the comparisons folder
        comparison_files = glob.glob("comparisons/comparison_*.txt")
        
        if not comparison_files:
            raise FileNotFoundError("No comparison reports found in comparisons/ folder")
        
        # Sort by modification time to get the latest
        latest_file = max(comparison_files, key=os.path.getmtime)
        
        print(f"📁 Found latest report: {latest_file}")
        
        # Read the file content
        with open(latest_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        return content
        
    except Exception as e:
        print(f"❌ Error finding latest comparison report: {e}")
        raise

def extract_comparison_insights(report_content: str) -> ComparisonInsights:
    """Extract key insights from the comparison report"""
    try:
        # Parse the report content to extract structured insights
        lines = report_content.split('\n')
        
        # Extract companies from the title
        companies = []
        for line in lines:
            if "COMPARATIVE ANALYSIS -" in line:
                companies_part = line.split("COMPARATIVE ANALYSIS - ")[1]
                # Remove the date/time if present
                if "=" in companies_part:
                    companies_part = companies_part.split("=")[0].strip()
                # Split by comma and clean up
                companies = [comp.strip() for comp in companies_part.split(',') if comp.strip()]
                break
        
        # Extract key findings and rankings
        key_findings = []
        rankings = []
        recommendations = ""
        
        current_section = None
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Identify sections
            if "Financial Performance Comparison" in line:
                current_section = "financial"
            elif "Ranking and Recommendation" in line:
                current_section = "ranking"
            elif "Investment Ranking" in line:
                current_section = "ranking"
            elif line.startswith(("1.", "2.", "3.", "4.", "5.", "6.")):
                # Extract numbered points as key findings
                if current_section != "ranking":
                    key_findings.append(line)
            elif line.startswith(("**", "-")) and current_section == "ranking":
                # Extract rankings
                rankings.append(line.replace("**", "").replace("-", "").strip())
        
        # Extract recommendations
        if rankings:
            recommendations = " ".join(rankings[:3])  # Top 3 recommendations
        
        # Extract generated date
        generated_date = ""
        for line in lines:
            if "Generated:" in line:
                generated_date = line.split("Generated: ")[1].strip()
                break
        
        insights = ComparisonInsights(
            companies=companies,
            key_findings=key_findings[:5],  # Top 5 findings
            rankings=rankings[:3],  # Top 3 rankings
            recommendations=recommendations,
            generated_date=generated_date
        )
        
        print(f"🔍 Extracted insights for: {', '.join(insights.companies)}")
        return insights
        
    except Exception as e:
        print(f"❌ Error extracting insights: {e}")
        # Return basic insights if parsing fails
        return ComparisonInsights(
            companies=["Unknown"],
            key_findings=["Analysis completed"],
            rankings=["1. Investment opportunity identified"],
            recommendations="Consider investment based on analysis",
            generated_date=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        )

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

# LinkedIn Article Generation Agents
research_writer_agent = Agent(
    name="research_writer_agent",
    instructions="""You are a professional financial content writer for LinkedIn.

Your task is to create engaging 220-word LinkedIn articles based on comparison insights and reviewer feedback.

Article Requirements:
- Target 200-240 words
- Professional LinkedIn format with emojis
- Data-driven insights from company comparisons
- Clear investment recommendations
- Appropriate hashtags (#Investment #StockAnalysis #Indonesia)
- Engaging opening hook
- Clear call-to-action
- Detailed analysis with supporting evidence

LinkedIn Format Example:
📊 COMPARATIVE ANALYSIS: [TICKERS]

[Opening hook with key finding]

🔍 Key Insights:
• [Point 1 with data and analysis]
• [Point 2 with data and analysis]
• [Point 3 with recommendation and evidence]
• [Point 4 with market context]
• [Point 5 with risk considerations]

💡 Investment Takeaway:
[Concluding recommendation with strategic insights]

#Investment #StockAnalysis #Indonesia

Revise based on reviewer feedback until approved. Make each iteration count!""",
    output_type=str
)

reviewer_agent = Agent(
    name="reviewer_agent",
    instructions="""You are a senior content editor for LinkedIn financial content.

Your task is to evaluate articles based on strict criteria:

Evaluation Criteria:
1. Word count (200-240 words)
2. Professional LinkedIn formatting with emojis
3. Accurate investment insights from data
4. Engaging tone and clarity
5. Clear call-to-action
6. Appropriate hashtags
7. No grammatical errors
8. Compelling opening hook
9. Detailed analysis with supporting evidence
10. Strategic insights and market context

Response Format:
- If approved: 'PASS: [return the EXACT full article content ready for posting]'
- If needs work: 'REJECT: [specific constructive feedback for improvement]'

CRITICAL: When approving, you MUST return the complete, formatted article content after 'PASS:'. Do not summarize or describe the article - return the actual LinkedIn post content.

Important Notes:
- You have maximum 3 rejection opportunities
- Use rejections wisely for meaningful improvements
- Be specific and actionable in feedback
- Focus on quality, depth, and engagement
- Ensure sufficient detail for 220-word target

After 3rd rejection, the article will be auto-passed regardless.""",
    output_type=str
)

# Menu and Utility Functions
def display_menu():
    """Display main menu and get user choice"""
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

# Service Functions
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

async def comparative_analysis():
    """Handle comparative analysis service with inline LinkedIn article option"""
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
            
            # Ask if user wants to generate LinkedIn article
            print("\n" + "-"*50)
            linkedin_choice = input("Would you like to generate a LinkedIn article from this comparison? (y/n): ").strip().lower()
            
            if linkedin_choice in ['y', 'yes']:
                print("\n🚀 Starting LinkedIn Article Generation...")
                await generate_linkedin_article()
            else:
                print("✅ Comparative analysis complete!")
            
        except Exception as e:
            print(f"❌ Error during analysis: {e}")

async def generate_linkedin_article():
    """Generate LinkedIn article using reviewer-controlled iterations"""
    try:
        # 1. Find and parse the latest comparison report
        print("📁 Finding latest comparison report...")
        latest_report = get_latest_comparison_report()
        insights = extract_comparison_insights(latest_report)
        
        # 2. Generate article with reviewer-controlled iterations
        current_draft = f"""Create a LinkedIn article based on these comparison insights:

Companies: {', '.join(insights.companies)}
Key Findings: {chr(10).join(insights.key_findings[:3])}
Rankings: {chr(10).join(insights.rankings[:2])}
Recommendations: {insights.recommendations}
Generated Date: {insights.generated_date}

Please create a professional LinkedIn article following the specified format."""
        
        iteration_count = 0
        
        while iteration_count < 3:
            iteration_count += 1
            print(f"📝 Writing iteration {iteration_count}/3...")
            
            # Writer creates/refines draft
            writer_result = await Runner.run(
                research_writer_agent, 
                current_draft
            )
            
            current_draft = writer_result.final_output
            print(f"🔍 Reviewer analyzing iteration {iteration_count}...")
            
            # Reviewer evaluates the draft
            review_prompt = f"""ARTICLE FOR REVIEW (Iteration {iteration_count}/3):

{current_draft}

Evaluate this article based on all criteria. If it meets standards and is ready for LinkedIn posting, respond with 'PASS: [final article]'. If it needs improvement, respond with 'REJECT: [specific constructive feedback for writer]'.

Remember: You have maximum 3 rejection opportunities. Use them wisely."""
            
            review_result = await Runner.run(
                reviewer_agent, 
                review_prompt
            )
            
            review_output = review_result.final_output
            
            # Check if reviewer passed the article
            if review_output.startswith("PASS:"):
                print("✅ Reviewer approved the article!")
                final_article = review_output.replace("PASS:", "").strip()
                break
            else:
                print(f"❌ Reviewer rejected iteration {iteration_count}")
                feedback = review_output.replace("REJECT:", "").strip()
                print(f"🔍 Reviewer Feedback (Iteration {iteration_count}):")
                print("-" * 50)
                print(feedback)
                print("-" * 50)
                current_draft = f"""PREVIOUS DRAFT:
{current_draft}

REVIEWER FEEDBACK (Iteration {iteration_count}):
{feedback}

PLEASE REVISE BASED ON THIS FEEDBACK:
Make the necessary improvements to address the reviewer's concerns."""
                
                # If this is the 3rd iteration, force pass
                if iteration_count == 3:
                    print("🔄 3rd iteration reached - auto-passing...")
                    final_article = current_draft
                    break
        
        # 3. Display the final article
        print("\n" + "="*60)
        print("📱 FINAL LINKEDIN ARTICLE")
        print("="*60)
        print(final_article)
        print("="*60)
        print("✅ Article ready for LinkedIn posting!")
        print(f"📊 Based on comparison of: {', '.join(insights.companies)}")
        print(f"📅 Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
    except Exception as e:
        print(f"❌ Error generating LinkedIn article: {e}")
        raise

# Main Application Loop
async def main():
    """Main application loop with streamlined services"""
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
