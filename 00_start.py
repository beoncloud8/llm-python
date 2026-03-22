from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

load_dotenv()
embeddings = OpenAIEmbeddings()
text = "Algoritma is a data science school based in Indonesia and Supertype is a data science consultancy with a distributed team of data and analytics engineers."
doc_embeddings = embeddings.embed_documents([text])

print(doc_embeddings)
