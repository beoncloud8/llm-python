import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
import streamlit as st

load_dotenv()

llm = ChatOpenAI(temperature=0, streaming=True, openai_api_key=os.getenv("OPENAI_API_KEY"))
search_tool = DuckDuckGoSearchRun()
tools = [search_tool]
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Use the search tool to find current information when needed."),
    ("placeholder", "{chat_history}"),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])
agent = create_agent(llm, tools)

# try: "what are the names of the kids of the 44th president of america"
# try: "top 3 largest shareholders of nvidia"
if prompt := st.chat_input():
    st.chat_message("user").write(prompt)
    with st.chat_message("assistant"):
        thinking_placeholder = st.empty()
        thinking_placeholder.write("🧠 thinking...")
        st_callback = StreamlitCallbackHandler(st.container())
        response = agent.invoke({"messages": [HumanMessage(content=prompt)]}, callbacks=[st_callback])
        thinking_placeholder.empty()  # Clear the thinking message
        # Extract and display the final answer
        if "messages" in response:
            messages = response["messages"]
            # Get the last AIMessage that has content
            for msg in reversed(messages):
                if hasattr(msg, 'content') and msg.content and msg.content.strip():
                    st.write(msg.content)
                    break
        else:
            st.write(str(response))
