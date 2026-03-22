"""
Obtain data from https://sectors.app
Accompanying course material: https://sectors.app/bulletin/ai-search
"""

import pandas as pd
from langchain_community.utilities import SQLDatabase
from sqlalchemy import create_engine
from langchain_groq import ChatGroq
from langchain_community.agent_toolkits import create_sql_agent, SQLDatabaseToolkit
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

if not os.path.exists('industry.db'):
    print("Creating industry.db")
    df = pd.read_csv('./datasets/industry-leaders-full.csv')
    engine = create_engine('sqlite:///industry.db')
    df.to_sql("industry", engine, index=False, if_exists='replace')
    db = SQLDatabase.from_uri("sqlite:///industry.db")
else:
    # connect to the existing database
    db = SQLDatabase.from_uri("sqlite:///industry.db")

print(db.get_usable_table_names())

# query = "SELECT * FROM industry WHERE sub_industry LIKE '%banks%'"
query2 = "SELECT * FROM industry WHERE total_market_cap > 1e14"
print(db.run(query2))

llm = ChatGroq(
    model="llama-3.3-70b-versatile"
)

# Create toolkit explicitly
toolkit = SQLDatabaseToolkit(db=db, llm=llm)

agent_executor = create_sql_agent(
    llm, 
    toolkit=toolkit, 
    agent_type="tool-calling", 
    verbose=True
)

agent_executor.invoke({
    "input": "Find companies in the coal industry by looking for 'Coal' in the sub_industry column. Show the company_name, market_cap_gainer_pct, and total_market_cap for coal companies. Order by market_cap_gainer_pct descending and return as a markdown table."
})

# Let's also run a direct query to see the coal companies
print("\n=== Direct Query for Coal Companies ===")
coal_query = "SELECT company_name, sub_industry, market_cap_gainer_pct FROM industry WHERE sub_industry LIKE '%Coal%' ORDER BY market_cap_gainer_pct DESC"
print(db.run(coal_query))