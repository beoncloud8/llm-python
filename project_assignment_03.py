# import os
import asyncio
import logging
import argparse
from typing import List, Optional, Dict, Any

from agents import Agent, Runner
from dotenv import load_dotenv

load_dotenv()

# Configure logging - suppress all HTTP logs by default
logging.basicConfig(level=logging.WARNING, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Suppress all HTTP/API logs
logging.getLogger("httpcore").setLevel(logging.CRITICAL)
logging.getLogger("httpx").setLevel(logging.CRITICAL)
logging.getLogger("openai").setLevel(logging.CRITICAL)
logging.getLogger("urllib3").setLevel(logging.CRITICAL)

# Configuration for agent sections
AGENT_CONFIGS: List[Dict[str, str]] = [
    {
        "name": "Mathematics Section",
        "field": "mathematics",
        "description": "mathematical questions"
    },
    {
        "name": "Physics Section", 
        "field": "physics",
        "description": "the physics science field"
    },
    {
        "name": "Literature Section",
        "field": "classical literature", 
        "description": "the classical literature field"
    },
    {
        "name": "Chemistry Section",
        "field": "chemistry",
        "description": "the chemistry science field"
    },
    {
        "name": "Biology Section",
        "field": "biology",
        "description": "the biology science field"
    }
]

#Now adding a 6th, 7th agent only requires a single new line in AGENT_CONFIGS. 
#Everything else — the dict, the handoffs list, and the routing instructions — updates automatically.

def create_section_agent(name: str, field: str, field_description: str) -> Agent:
    """Create a specialized section agent with consistent instructions.
    
    Args:
        name: The display name for the agent
        field: The academic field (e.g., 'mathematics', 'physics')
        field_description: Description of the field for instructions
        
    Returns:
        Configured Agent instance
    """
    return Agent(
        name=name,
        instructions=f"""You are a {field} professor. Your purpose is to handle questions related to {field_description}.
        Before answering the question, introduce yourself as {{name}}. Make sure to verify the accuracy of the question.
        If the question is not clear, ask for more information. Be concise, limit your answer to 150 words that focus on the key points.
        Use bullet points to organize your answer."""
    )

# Create specialized section agents using configuration
agents: Dict[str, Agent] = {
    config["field"]: create_section_agent(
        name=config["name"],
        field=config["field"],
        field_description=config["description"]
    )
    for config in AGENT_CONFIGS  # dict comprehension, cleaner than a for-loop
}

# Auto-generate routing instructions from config
routing_rules = "\n".join(
    f"If the question is related to {c['field']}, allocate it to the {c['name']}."
    for c in AGENT_CONFIGS
)

query_perceptor = Agent(
    name="Query Perceptor",
    instructions=f"""You are a question perceptor. Analyze the user's question and 
route it to the appropriate section agent.
{routing_rules}
If the question is not clear, ask for more information.""",
    handoffs=list(agents.values()),
)

async def process_query(query: str) -> Optional[str]:
    """Process a user query through the agent system.
    
    Args:
        query: The user's question or query
        
    Returns:
        The agent's response or None if an error occurs
    """
    try:
        result = await Runner.run(query_perceptor, query)
        return result.final_output
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        return None

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments.
    
    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(description="AI Query Processing System")
    parser.add_argument(
        "--query", 
        type=str, 
        help="Query to process (if not provided, will prompt for input)"
    )
    parser.add_argument(
        "--verbose", 
        action="store_true", 
        help="Enable verbose logging (for debugging)"
    )
    return parser.parse_args()

async def main(query: Optional[str] = None) -> None:
    """Main function to run the query processing system.
    
    Args:
        query: Optional query string. If None, will prompt user for input.
    """
    if query is None:
        query = input("Enter your query: ")
    
    if not query.strip():
        logger.warning("Empty query provided")
        print("Please provide a valid query.")
        return
    
    response = await process_query(query)
    
    if response:
        print(f"😊: {query}")
        print(f"🤖: {response}")
    else:
        print("Sorry, I encountered an error while processing your query. Please try again.")

if __name__ == "__main__":
    args = parse_arguments()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    asyncio.run(main(query=args.query))