
from langchain_ollama import OllamaLLM
from dotenv import load_dotenv
load_dotenv()

# Depreciated
# # from langchain.chains import RetrievalQA, ConversationalRetrievalChain
# # from langchain_community.llms import Ollama
# llm1 = Ollama(
#     model= "tinyllama",
#     temperature=0.1
# )

# Use Ollama LLM
llm = OllamaLLM(
    model= "tinyllama",
    temperature=0.1
)

response = llm.invoke("What is the capital of Pakistan?")
print(response)