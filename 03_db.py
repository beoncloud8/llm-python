from dotenv import load_dotenv
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_openai import OpenAI
import sqlite3

load_dotenv()

# Create a simple sample database
def create_sample_db():
    conn = sqlite3.connect('sample.db')
    cursor = conn.cursor()
    
    # Create responses table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS responses (
            id INTEGER PRIMARY KEY,
            name TEXT,
            email TEXT,
            country TEXT,
            hear_about_us TEXT,
            response_text TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Insert sample data
    sample_data = [
        (1, 'John Doe', 'john@email.com', 'USA', 'Social Media', 'Great experience!', '2024-01-01'),
        (2, 'Jane Smith', 'jane@email.com', 'Canada', 'Friend', 'Very helpful', '2024-01-02'),
        (3, 'Bob Johnson', 'bob@email.com', 'UK', 'Google', 'Excellent service', '2024-01-03'),
        (4, 'Alice Brown', 'alice@email.com', 'USA', 'Social Media', 'Would recommend', '2024-01-04'),
        (5, 'Charlie Wilson', 'charlie@email.com', 'Australia', 'Friend', 'Amazing!', '2024-01-05')
    ]
    
    cursor.executemany('INSERT OR REPLACE INTO responses VALUES (?,?,?,?,?,?,?)', sample_data)
    conn.commit()
    conn.close()

create_sample_db()

# Use the created database
dburi = "sqlite:///sample.db"
db = SQLDatabase.from_uri(dburi)
llm = OpenAI(temperature=0)

# Simple manual SQL generation and execution
def ask_database(question):
    print(f"\n{'='*60}")
    print(f"📝 Question: {question}")
    print('='*60)
    
    # Generate SQL using LLM
    prompt = f"""Given this database schema:
{db.get_table_info()}

Convert this question to SQL for SQLite database: {question}
Return only the SQL query, no explanation.
Note: Use PRAGMA table_info(responses) for table description, not DESCRIBE."""
    
    sql_query = llm.invoke(prompt).strip()
    print(f"🔍 Generated SQL:\n{sql_query}")
    
    # Execute query
    try:
        result = db.run(sql_query)
        print(f"✅ Results:")
        
        # Format results for better readability
        if isinstance(result, list) and result:
            if len(result) == 1 and len(result[0]) == 1:
                # Single value result
                print(f"   {result[0][0]}")
            elif all(isinstance(row, tuple) for row in result):
                # Table-like results
                for row in result:
                    if len(row) == 2:
                        print(f"   {row[0]}: {row[1]}")
                    else:
                        print(f"   {row}")
            else:
                print(f"   {result}")
        elif result:
            print(f"   {result}")
        else:
            print("   No results found")
            
    except Exception as e:
        print(f"❌ Error executing query: {e}")
        return None

# Test queries
questions = [
    "How many rows is in the responses table of this db?",
    "Describe the responses table",
    "What are the top 3 countries where these responses are from?",
    "Give me a summary of how these customers come to hear about us. What is the most common way they hear about us?"
]

for question in questions:
    ask_database(question)
