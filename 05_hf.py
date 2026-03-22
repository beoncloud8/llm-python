from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate

load_dotenv()

hub_llm = HuggingFaceEndpoint(repo_id="mrm8488/t5-base-finetuned-wikiSQL")

# prompt = PromptTemplate.from_template(
#     "Translate English to SQL: {question}"    
# )

# result = hub_llm.invoke(prompt.format(question="What is the average age of the respondents using a mobile device?"))
# print(result)


# second example below:
hub_llm = HuggingFaceEndpoint(
    repo_id='distilgpt2',
    task="text-generation",
    temperature=0.7,
    max_new_tokens=100
)

prompt = PromptTemplate.from_template(
    "You had one job ! You're the {profession} and you didn't have to be sarcastic"
)

# Direct invocation with HuggingFaceEndpoint
print(hub_llm.invoke(prompt.format(profession="customer service agent")))
print(hub_llm.invoke(prompt.format(profession="politician")))
print(hub_llm.invoke(prompt.format(profession="Fintech CEO")))
print(hub_llm.invoke(prompt.format(profession="insurance agent")))