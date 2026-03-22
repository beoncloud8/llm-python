import time
from dotenv import load_dotenv
import langchain
from langchain_openai import OpenAI
from langchain_community.callbacks import get_openai_callback
from langchain_core.caches import InMemoryCache
from langchain_core.globals import set_llm_cache

load_dotenv()

# to make caching obvious, we use a slower model
llm = OpenAI(model="gpt-3.5-turbo-instruct")

set_llm_cache(InMemoryCache())

with get_openai_callback() as cb:
    start = time.time()
    result = llm.invoke("What doesn't fall far from the tree?")
    print(result)
    end = time.time()
    print("--- cb")
    print(str(cb) + f" ({end - start:.2f} seconds)")

with get_openai_callback() as cb2:
    start = time.time()
    result2 = llm.invoke("What doesn't fall far from the tree?")
    result3 = llm.invoke("What doesn't fall far from the tree?")
    end = time.time()
    print(result2)
    print(result3)
    print("--- cb2")
    print(str(cb2) + f" ({end - start:.2f} seconds)")
