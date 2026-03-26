import streamlit as st
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_ollama import ChatOllama

# --- 1. Setup & Configuration ---
load_dotenv()

# Assuming your tools are in a local file named tools.py
try:
    from tools import search_tool, wiki_tool, save_tool
    tools = [search_tool, wiki_tool, save_tool]
except ImportError:
    st.error("Could not find 'tools.py'. Please ensure your tool definitions are accessible.")
    st.stop()

# Page Config
st.set_page_config(page_title="AI Research Assistant", page_icon="🔬")

# Define Structured Output Schema
class ResearchResponse(BaseModel):
    title: str = Field(description="The title of the research topic")
    content: str = Field(description="The main research findings")
    text: str = Field(description="Additional supporting text or data")
    research_type: str = Field(description="The type of research conducted")
    category: str = Field(description="Classification category")

# Initialize LLM and Parser
chat_model = ChatOllama(model="llama3.2") 
parser = PydanticOutputParser(pydantic_object=ResearchResponse)

# Define Prompt Template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a research assistant. Use tools to find info. Format output as JSON.\n{format_instructions}"),
    ("placeholder", "{chat_history}"),
    ("human", "{query}"),
    ("placeholder", "{agent_scratchpad}"),
]).partial(format_instructions=parser.get_format_instructions())

# Initialize Agent
agent = create_tool_calling_agent(llm=chat_model, prompt=prompt, tools=tools)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# --- 2. Streamlit UI Layout ---
st.title("🔬 AI Research Assistant")
st.markdown("Enter a topic below to generate a structured research summary using Llama 3.2.")

query = st.text_input("What would you like to research?", placeholder="e.g., The impact of quantum computing on cybersecurity")

if st.button("Start Research"):
    if query:
        with st.spinner("Searching and synthesizing information..."):
            try:
                # Run the Agent
                raw_response = agent_executor.invoke({"query": query, "chat_history": []})
                output_str = raw_response.get("output")
                
                # Parse the output into our Pydantic model
                structured_data = parser.parse(output_str)

                # --- 3. Display Results ---
                st.divider()
                st.header(f"Results: {structured_data.title}")
                
                # Use columns for metadata
                col1, col2 = st.columns(2)
                with col1:
                    st.info(f"**Category:** {structured_data.category}")
                with col2:
                    st.info(f"**Type:** {structured_data.research_type}")

                st.subheader("Key Findings")
                st.write(structured_data.content)

                if structured_data.text:
                    with st.expander("View Additional Details"):
                        st.write(structured_data.text)
                
                st.success("Research completed successfully!")

            except Exception as e:
                st.error(f"An error occurred during processing.")
                with st.expander("Show Technical Error"):
                    st.write(str(e))
                    st.write("Raw LLM Output:", raw_response.get("output") if 'raw_response' in locals() else "No output.")
    else:
        st.warning("Please enter a query first.")

# Footer
st.sidebar.markdown("---")
st.sidebar.caption("Powered by LangChain + Ollama + Streamlit")