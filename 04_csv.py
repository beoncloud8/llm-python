from dotenv import load_dotenv
from langchain_openai import OpenAI 
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_experimental.agents import create_csv_agent
import pandas as pd

load_dotenv()

filepath = "academy.csv"
loader = CSVLoader(filepath)
data = loader.load()

# Display data in a readable format
print(f"Loaded {len(data)} rows from {filepath}")
print("\nFirst 3 rows:")
for i, doc in enumerate(data[:3]):
    print(f"\nRow {i+1}:")
    print(doc.page_content)

llm = OpenAI(temperature=0)

agent = create_csv_agent(llm, filepath, verbose=True, allow_dangerous_code=True)
print("\n" + "="*60)
print("ANALYTICAL QUESTIONS:")
print("="*60)

agent.run("What percentage of the respondents are students versus professionals?")
agent.run("List the top 3 devices that the respondents use to submit their responses")
agent.run("Consider iOS and Android as mobile devices. What is the percentage of respondents that discovered us through social media submitting this from a mobile device?")
