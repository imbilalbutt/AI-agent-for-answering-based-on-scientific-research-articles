from langchain import hub
from langchain.agents import create_react_agent, AgentExecutor
from tools import *
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini")

# Pull the ReAct prompt (a standard agent prompt)
prompt = hub.pull("hwchase17/react")

#  Agent 2
hubspot_tools = [get_contact, add_contact]
agent = create_react_agent(llm, hubspot_tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=hubspot_tools, verbose=True)

# Example: Create a contact
response = agent_executor.invoke({
    "input": "Create a new contact in HubSpot with email john@example.com, first name John, last name Doe."
})
print(response["output"])