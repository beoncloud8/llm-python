import asyncio
import os
from dotenv import load_dotenv
from typing import List, Dict, Any
from pydantic import BaseModel
from dataclasses import dataclass
from typing import Literal
import json
from agents import Agent, ItemHelpers, Runner, TResponseInputItem, trace, function_tool, AgentOutputSchema
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

@function_tool
def read_from_txt(filename: str = "3_research_notes.txt") -> str:
    """
    Read company data from a text file in the output folder
    """
    try:
        file_path = os.path.join("output", filename)
        if not os.path.exists(file_path):
            return f"File {filename} not found in output folder. Please ensure the file exists."
        
        with open(file_path, "r") as file:
            data = file.read()
            if not data.strip():
                return f"File {filename} is empty. No data to summarize."
            return data
    except Exception as e:
        return f"Error reading file {filename}: {str(e)}"
    
secretary = Agent(
    name="secretary",
    instructions=(
        "Given a company name or ticker by the user, read the company data from a text file in the output folder."
        "Use the read_from_txt function to read the data. The default file is '3_research_notes.txt' but you can specify other files."
        "Summarize the data into a comprehensive Markdown table including all relevant company information."
        "If there is any feedback from the evaluator, incorporate it to improve the report in subsequent iterations."
        "Ensure the summary includes: company basics, financial metrics, performance data, contact info, and business context."
    ),
    tools=[read_from_txt] 
)

@dataclass
class EvaluationFeedback:
    feedback: str
    score: Literal["pass", "expect_improvement", "fail"]

rubric = """
You are an evaluator that assesses the quality of a company overview summary based on the following criteria:
- [] includes clear company name and stock ticker (symbol)
- [] provides a concise company description (industry, listing board, other background)
- [] compliant Markdown table format
- [] includes key financial metrics (market cap, stock price, employees)
- [] contains recent performance data and price ranges
- [] provides contact information and website
- [] includes relevant business context and notes

7 criteria met: score = "pass"
5-6 criteria met: score = "expect_improvement"
4 or fewer criteria met: score = "fail"

Be specific in your feedback on what rubric points were missed and how to improve the content quality.
"""

evaluator = Agent[None](
    name="evaluator", 
    instructions=rubric,
    output_type=EvaluationFeedback, 
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
    3. Write all research results to a file inside the "output" folder
    4. Use "3_research_notes.txt" as the filename
    
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
    
    tickers_list = []  # Initialize to avoid scope issues

    # Ensure the entire workflow is a single trace
    with trace("Sequential research flow"):
        """
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
        """

        # 3. Judge Critic Pattern
        msg = input(f"🤖: What company are you interested in ? \n👧: ")
        # if msg == "n":
        #     break
        
        input_items: list[TResponseInputItem] = [{"content": msg, "role": "user"}]
        
        max_attempts = 3
        attempt = 0
        best_summary = None
        
        while True:
            attempt += 1
            if attempt > max_attempts:
                if best_summary:
                    summary_text = ItemHelpers.text_message_outputs(best_summary.new_items)
                    print(f"Reached maximum attempts ({max_attempts}). Using best attempt:")
                    print(summary_text)
                else:
                    print(f"Reached maximum attempts ({max_attempts}) without generating a valid summary.")
                break

            # 3.1 The Secretary Agent reads the research notes and summarizes them into a Markdown table
            summarized_results = await Runner.run(
                secretary,
                f"Summarize the research notes for {msg}"
            )

            # Update input_items to include summarizer's response (for chaining)
            input_items = summarized_results.to_input_list()
            
            print(f"Secretary - overview summary generated. \nIteration {attempt}")

            # 3.2 Run evaluator agent
            evaluator_result = await Runner.run(evaluator, input_items)
            result: EvaluationFeedback = evaluator_result.final_output

            print(f"Evaluator score: {result.score}")
            if result.feedback:
                print(f"Feedback: {result.feedback}")

            # Store the best summary (highest scoring)
            if result.score in ["pass", "expect_improvement"]:
                best_summary = summarized_results

            # 3.3 Check Evaluation Result
            if result.score == "pass":
                print("The stock summary is 💡 good enough, exiting.")
                # Extract the plain-text summary from the summarizer output
                summary_text = ItemHelpers.text_message_outputs(summarized_results.new_items)
                print(f"Final summary:\n{summary_text}")
                break

            # If not passing and attempts remain, attach feedback for next round
            print("Re-running with feedback")
            input_items.append({
                "role": "user",
                "content": f"Please improve the summary based on this feedback: {result.feedback}"
            })

        # 4. Open the sectors.app pages for the company (if we have tickers_list)
        if tickers_list:  # Check if we have tickers from the screener
            open_sectors_app = input("Do you want to open the sectors.app page for the last company on this list too? (y/n): ").lower()
            if open_sectors_app == 'y':
                last_ticker = tickers_list[-1]  
                await Runner.run(
                    browser_agent,
                    f"Open sectors.app page for {last_ticker}"
                )
                print(f"I have opened the sectors.app search and company pages for the ticker {last_ticker} in your web browser. You can now review financial data, news, and analysis related to {last_ticker} directly on those pages. If you need a summary or specific information from those pages, let me know!")
        else:
            print("Done! No ticker list available from screener.")

if __name__ == "__main__":
    asyncio.run(main())