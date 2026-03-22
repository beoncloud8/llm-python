from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import InMemoryVectorStore
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAI

load_dotenv()
embeddings = OpenAIEmbeddings()

loader = TextLoader('news/result.txt')

documents = loader.load()
print(len(documents))
text_splitter = CharacterTextSplitter(chunk_size=2500, chunk_overlap=0)
texts = text_splitter.split_documents(documents)
# print(texts)

docsearch = InMemoryVectorStore.from_documents(texts, embeddings)
template = """Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Keep the answer concise and relevant to the question.

Context: {context}

Question: {question}

Helpful Answer:"""
prompt = PromptTemplate.from_template(template)

qa_chain = (
    {"context": docsearch.as_retriever(), "question": RunnablePassthrough()}
    | prompt
    | OpenAI()
    | StrOutputParser()
)

def query(q):
    print("Query: ", q)
    print("Answer: ", qa_chain.invoke(q))

query("What are the effects of legislations surrounding emissions on the Australia coal market?")
query("What are China's plans with renewable energy?")
query("Is there an export ban on Coal in Indonesia? Why?")
query("Who are the main exporters of Coal to China? What is the role of Indonesia in this?")