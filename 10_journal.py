from dotenv import load_dotenv
load_dotenv()

import argparse
import logging
import sys
from pathlib import Path
import os
from llama_index.core import (
    Document,
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage,
    SimpleDirectoryReader
)
from llama_index.core.node_parser import SimpleNodeParser

# to see token counter and token usage for the LLM and Embedding
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))


OBSIDIAN_DIR = os.getenv("OBSIDIAN_JOURNAL_DIR", "/Users/beoncloud/Documents/bcvault/Journal")

# Validate directory exists
if not os.path.exists(OBSIDIAN_DIR):
    raise FileNotFoundError(f"Obsidian directory not found: {OBSIDIAN_DIR}")

docs = SimpleDirectoryReader(OBSIDIAN_DIR).load_data()


def read_journal_md(file_path):
    from bs4 import BeautifulSoup
    import markdown
    import re
    
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
            
        if not text.strip():
            print(f"Warning: Empty file {file_path}")
            return ""
            
        html = markdown.markdown(text)
        soup = BeautifulSoup(html, "html.parser")

        # Get all paragraph content
        ps = soup.find_all("p")
        
        if not ps:
            print(f"Warning: No paragraphs found in {file_path}")
            return ""
            
        # Combine all paragraph text
        result = " ".join([p.text for p in ps])

        print(f"Finished processing {file_path}")
        return result
        
    except FileNotFoundError:
        print(f"Error: File not found {file_path}")
        return ""
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return ""


def create_journal_nodes(dir_path):
    """
    Examples: https://gpt-index.readthedocs.io/en/stable/guides/primer/usage_pattern.html
    """
    docs = []
    parser = SimpleNodeParser()

    # loop through each markdown file in the directory
    try:
        for file_path in Path(dir_path).glob("*.md"):
            md = read_journal_md(file_path)
            if md.strip():  # Only add non-empty documents
                # construct documents manually using the lower level Document struct
                docs.append(Document(text=md))
            
        if not docs:
            print("Warning: No valid documents found")
            return [], []
            
        nodes = parser.get_nodes_from_documents(docs)
        return nodes, docs
        
    except Exception as e:
        print(f"Error processing directory {dir_path}: {str(e)}")
        return [], []

if Path("./storage").exists():
    try:
        storage_context = StorageContext.from_defaults(persist_dir="./storage")
        index = load_index_from_storage(storage_context)
        print("Loaded existing index from storage")
    except Exception as e:
        print(f"Error loading index from storage: {str(e)}")
        print("Creating new index...")
        nodes, docs = create_journal_nodes(OBSIDIAN_DIR)
        if nodes:
            index = VectorStoreIndex(nodes)
            index.storage_context.persist(persist_dir="./storage")
            print("Created and persisted new index")
        else:
            print("No nodes to create index")
            exit(1)
else:
    nodes, docs = create_journal_nodes(OBSIDIAN_DIR)
    if nodes:
        index = VectorStoreIndex(nodes)
        index.storage_context.persist(persist_dir="./storage")
        print("Created and persisted new index")
    else:
        print("No nodes to create index")
        exit(1)

if __name__ == "__main__":
    """
    Usage: python 10_journal_x.py -q "what are places I ate at in March and April?"
    """
    query_engine = index.as_query_engine()
    # cli argument parser
    parser = argparse.ArgumentParser(
        prog="QueryJournal",
        description="Query my bullet journals in Obsidian using Llama Index."
    )
    parser.add_argument(
        "-q",
        "--query",
        type=str,
        help="Ask a question answerable in my journals",
        required=True
    )
    args = parser.parse_args()
    query = args.query

    if(query):
        res = query_engine.query(query)
        print(f"Query: {query}")
        print(f"Results: \n {res}")
    else:
        print("No query provided. Exiting...")
        exit(0)
