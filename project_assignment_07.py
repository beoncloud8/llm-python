"""
LinkedIn Article Generation with Judge-Critic Pattern
Creates 150-word LinkedIn articles from latest comparison reports using reviewer-controlled iterations.

# Usage:
python project_assignment_07.py
"""

import asyncio
import os
import glob
from datetime import datetime
from typing import List, Dict, Any
from pydantic import BaseModel
from agents import Agent, Runner, trace, function_tool
from dotenv import load_dotenv

load_dotenv()

# Data Models
class ComparisonInsights(BaseModel):
    """Model for comparison insights extracted from reports"""
    companies: List[str]
    key_findings: List[str]
    rankings: List[str]
    recommendations: str
    generated_date: str

# Report Discovery and Parsing Functions
def get_latest_comparison_report() -> str:
    """Find the latest comparison report in the comparisons folder"""
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

# Main Workflow Functions
async def generate_linkedin_article():
    """Generate LinkedIn article using reviewer-controlled iterations"""
    try:
        print("🚀 Starting LinkedIn Article Generation...")
        
        # 1. Find and parse latest comparison report
        print("📁 Finding latest comparison report...")
        latest_report = get_latest_comparison_report()
        insights = extract_comparison_insights(latest_report)
        
        # 2. Generate article with reviewer-controlled iterations
        with trace("Judge_Critic_Workflow_Reviewer_Controlled"):
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
            
            # 3. Display final article
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

async def main():
    """Main application entry point"""
    print("\n" + "="*60)
    print("📱 LinkedIn Article Generator")
    print("Judge-Critic Pattern with Reviewer-Controlled Iterations")
    print("="*60)
    
    try:
        await generate_linkedin_article()
    except Exception as e:
        print(f"❌ System error: {e}")
        print("\n💡 Please ensure:")
        print("   1. Comparison reports exist in 'comparisons/' folder")
        print("   2. API keys are properly configured")
        print("   3. Network connection is stable")

if __name__ == "__main__":
    asyncio.run(main())
