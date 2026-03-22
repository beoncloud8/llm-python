import time
from dotenv import load_dotenv
import langchain
from langchain_openai import OpenAI
from langchain_community.callbacks import get_openai_callback

from langchain_text_splitters import CharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.cache import SQLiteCache

load_dotenv()

# Configure cache for the specific LLM
llm = OpenAI(model="gpt-3.5-turbo-instruct", cache=SQLiteCache(database_path=".langchain.db"))
no_cache_llm = OpenAI(model="gpt-3.5-turbo-instruct", cache=None)

text_splitter = CharacterTextSplitter()

with open("news/summary.txt") as f:
    news = f.read()

texts = text_splitter.split_text(news)
print(texts)

docs = [Document(page_content=t) for t in texts[:3]]

# Create a simple summarization chain using LCEL
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

summarize_prompt = ChatPromptTemplate.from_template("Summarize the following text:\n\n{text}")
chain = summarize_prompt | llm | StrOutputParser()

with get_openai_callback() as cb:
    start = time.time()
    result = chain.invoke({"text": "\n".join([doc.page_content for doc in docs])})
    end = time.time()
    print("--- result1")
    print(result)
    print(str(cb) + f" ({end - start:.2f} seconds)")


with get_openai_callback() as cb2:
    start = time.time()
    result = chain.invoke({"text": "\n".join([doc.page_content for doc in docs])})
    end = time.time()
    print("--- result2")
    print(result)
    print(str(cb2) + f" ({end - start:.2f} seconds)")
