from transformers import pipeline
from langchain_huggingface import HuggingFacePipeline
from dotenv import load_dotenv
load_dotenv()

# Depreciated
# # from langchain.embeddings import HuggingFaceEmbeddings
# # from langchain.chains import RetrievalQA, ConversationalRetrievalChain
# # # from langchain.llms import HuggingFacePipeline


hf_pipeline = pipeline(
    "text-generation",
    model="google/flan-t5-small",
    max_length=512
)
llm = HuggingFacePipeline(pipeline=hf_pipeline)

response = llm.invoke("What is the capital of Pakistan?")
print(response)