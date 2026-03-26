from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_tool_calling_agent, AgentExecutor, create_react_agent
from tools import search_tool, wiki_tool, save_tool
import torch

load_dotenv()

from transformers import pipeline
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline

# model_name = "google/flan-t5-small"

# hf_pipeline = pipeline(
#     "text2text-generation",
#     model=model_name,
#     max_length=512
# )
# llm = HuggingFacePipeline(pipeline=hf_pipeline)
# chat_model = ChatHuggingFace(llm=llm)


model_name = "Qwen/Qwen2.5-7B-Instruct"  # or your chosen model

hf_pipeline = pipeline(
    "text-generation",
    model=model_name,
    tokenizer=model_name,
    max_new_tokens=1024,        # use max_new_tokens, not max_length
    device_map="auto",          # uses GPU if available, falls back to CPU
    torch_dtype=torch.float16,  # reduces memory usage
)

llm = HuggingFacePipeline(pipeline=hf_pipeline)
chat_model = ChatHuggingFace(llm=llm)


class ResearchResponse(BaseModel):
    topic: str
    summary: str
    sources: list[str]
    tools_used: list[str]


parser = PydanticOutputParser(pydantic_object=ResearchResponse)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are a research assistant that will help generate a research paper.
            Answer the user query and use neccessary tools. 
            
            You have access to the following tools:
            {tools}
            
            Use the following format:
            Question: the input question you must answer
            Thought: you should always think about what to do
            Action: the action to take, should be one of [{tool_names}]
            Action Input: the input to the action
            Observation: the result of the action
            ... (this Thought/Action/Action Input/Observation can repeat N times)
            Thought: I now know the final answer
            Final Answer: the final answer to the original input question
            
            Wrap the final output in this format and provide no other text\n{format_instructions}
            """,
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{query}"),
        ("{agent_scratchpad}"),
    ]
).partial(format_instructions=parser.get_format_instructions())

tools = [search_tool, wiki_tool, save_tool]
agent = create_react_agent( #create_tool_calling_agent(
    # llm = llm,
    llm=chat_model,
    prompt = prompt,
    tools = tools
)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
query = input("What can I help you research?")
# raw_response = agent_executor.invoke({"query" : "What is the capital of Pakistan?"})


try:
    raw_response = agent_executor.invoke({"query": query, "chat_history": []})
    try:
        structured_response = parser.parse(raw_response.get("output"))
        print(structured_response)
    except Exception as e:
        print(f"Error parsing response: {e}")
        print(f"Raw response: {raw_response.get('output')}")
except Exception as e:
    print(f"Error during agent execution: {e}")