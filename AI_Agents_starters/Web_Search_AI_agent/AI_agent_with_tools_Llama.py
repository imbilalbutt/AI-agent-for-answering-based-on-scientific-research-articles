from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_tool_calling_agent, AgentExecutor
from tools import search_tool, wiki_tool, save_tool

load_dotenv()

from langchain_ollama import ChatOllama
chat_model = ChatOllama(model="llama3.2") 

class ResearchResponse(BaseModel):
    title: str
    content: str
    text: str
    research_type: str = Field(description="The type of research conducted")
    category: str

    def get_content(self):
        return self.content

    def get_text(self):
        return self.text

    def get_title(self):
        return self.title

    def get_type(self):
        return self._type

    def get_category(self):
        return self.category


parser = PydanticOutputParser(pydantic_object=ResearchResponse)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are a research assistant that will help generate a research paper.
            Answer the user query and use neccessary tools. 
            Wrap the output in this format and provide no other text\n{format_instructions}
            """,
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{query}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
).partial(format_instructions=parser.get_format_instructions())

tools = [search_tool, wiki_tool, save_tool]
agent = create_tool_calling_agent(
    # llm = llm,
    llm=chat_model,
    prompt = prompt,
    tools = tools
)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
query = input("What can I help you research? ")
raw_response = agent_executor.invoke({"query": query, "chat_history": []})
# raw_response = agent_executor.invoke({"query" : "What is the capital of Pakistan?"})


try:
    structured_response = parser.parse(raw_response.get("output"))
    # Access fields directly using dot notation
    print("Title:", structured_response.title)
    print("Content:", structured_response.content)

except Exception as e:
    print("Error parsing response:", e)
    print("Raw response:", raw_response)

