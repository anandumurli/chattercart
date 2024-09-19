from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain.tools.retriever import create_retriever_tool
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

import os

load_dotenv(override=True)
API_KEY=os.getenv("OPEN_API_KEY")

# Making Data Folder
DATA_PATH = "data/"
os.makedirs(DATA_PATH, exist_ok=True)

# Function to list files in the data folder
def list_files():
    return os.listdir(DATA_PATH)

files = list_files()
context_file_name = DATA_PATH + files[0]
# we take the first file in consideration always, might change this functionality later

#LOADING FILE
# loader = DirectoryLoader(DATA_PATH)
documents = TextLoader(context_file_name).load()

# SPLITING FILE
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(documents)

# VECTORSTORE
vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings(openai_api_key=API_KEY))
retriever = vectorstore.as_retriever()

# specifying tools for the agent
tavily_search = TavilySearchResults(
max_results=5,
search_depth="advanced",
)

# converting the retriever to get information from the context (the RAG retriever)
retriever_tool = create_retriever_tool(
    retriever,
    'text_retreiver',
    "Give answer based on the file",
)

#Declaring tools for our agent
tools = [retriever_tool, tavily_search]

# setting the llm and a memory variable this is using 
# a function that is built in which can store memories 
# agaist some session id, we have defined it as test-session
# it's just a name
llm = ChatOpenAI(api_key=API_KEY, model="gpt-3.5-turbo")
memory = InMemoryChatMessageHistory(session_id="test-session")

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant for the business owner. Most questions asked will be about the business itself or something apart from the context provided. Keep the answers concise upto three or four lines whenever possible. You will answer in behalf of the owner."),
        # First put the history placeholder
        ("placeholder", "{chat_history}"),
        # Then the new input placeholder
        ("human", "{input}"),
        # Finally the scratchpad placeholder
        ("placeholder", "{agent_scratchpad}"),
    ]
)
# creating an agent
agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools)

# finally creating the runnable which uses the agent to chat 
# with us.
agent_with_chat_history = RunnableWithMessageHistory(
    agent_executor,
    # This is needed because in most real world scenarios, a session id is needed
    # It isn't really used here because we are using a simple in memory ChatMessageHistory
    lambda session_id: memory,
    input_messages_key="input",
    history_messages_key="chat_history",
)

# creating the config which will pass for the chat-history 
# through the lambda function which passed to memory to place against
# and confirm
config = {"configurable": {"session_id": "test-session"}}

while(True):
    response =  agent_with_chat_history.invoke({"input": input("Ask your question!!")}, config)["output"]
    print(response)