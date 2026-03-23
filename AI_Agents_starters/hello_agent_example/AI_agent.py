from langchain import hub
from langchain.agents import create_react_agent, AgentExecutor
from tools import *
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini")


# Pull the ReAct prompt (a standard agent prompt)
prompt = hub.pull("hwchase17/react")

# Create the agent
tools = [get_current_time]

agent = create_react_agent(llm, tools, prompt)

# AgentExecutor runs the agent
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# Try it
result = agent_executor.invoke({"input": "What time is it right now?"})
print(result["output"])