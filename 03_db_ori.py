from dotenv import load_dotenv
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_openai import OpenAI
import sqlite3
import os

load_dotenv()

# Create academy database with sample data
def create_academy_db():
    conn = sqlite3.connect('academy/academy.db')
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
            course_taken TEXT,
            rating INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Insert sample academy data
    sample_data = [
        (1, 'John Doe', 'john@email.com', 'USA', 'Social Media', 'Great Python course!', 'Python Basics', 5, '2024-01-01'),
        (2, 'Jane Smith', 'jane@email.com', 'Canada', 'Friend', 'Very helpful ML workshop', 'Machine Learning', 4, '2024-01-02'),
        (3, 'Bob Johnson', 'bob@email.com', 'UK', 'Google', 'Excellent data science program', 'Data Science', 5, '2024-01-03'),
        (4, 'Alice Brown', 'alice@email.com', 'USA', 'Social Media', 'Would recommend web dev', 'Web Development', 4, '2024-01-04'),
        (5, 'Charlie Wilson', 'charlie@email.com', 'Australia', 'Friend', 'Amazing AI course!', 'AI Fundamentals', 5, '2024-01-05'),
        (6, 'Diana Prince', 'diana@email.com', 'Germany', 'LinkedIn', 'Good database course', 'SQL Mastery', 3, '2024-01-06'),
        (7, 'Eve Adams', 'eve@email.com', 'France', 'YouTube', 'Learned a lot about deep learning', 'Deep Learning', 4, '2024-01-07'),
        (8, 'Frank Miller', 'frank@email.com', 'Japan', 'Twitter', 'Best cloud computing course', 'Cloud Computing', 5, '2024-01-08')
    ]
    
    cursor.executemany('INSERT OR REPLACE INTO responses VALUES (?,?,?,?,?,?,?,?,?)', sample_data)
    conn.commit()
    conn.close()
    print("Academy database created with 8 sample responses.")

# Create the database
create_academy_db()

# Use the academy database
dburi = "sqlite:///academy/academy.db"
db = SQLDatabase.from_uri(dburi)
llm = OpenAI(temperature=0)

# Manual SQL generation and execution (modern approach)
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
    "Give me a summary of how these customers come to hear about us. What is the most common way they hear about us?",
    "What is the average rating for all courses?",
    "Which course has the highest number of students?",
    "Show me students who gave a rating of 5 or higher"
]

for question in questions:
    ask_database(question)