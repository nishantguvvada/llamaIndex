from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.llms.gemini import Gemini
import os
from dotenv import load_dotenv

load_dotenv()

documents = SimpleDirectoryReader("db").load_data()
index = VectorStoreIndex.from_documents(documents)
# llm = Gemini(model="models/gemini-2.0-flash", api_key=os.getenv('GEMINI_API_KEY'))
query_engine = index.as_query_engine(llm="default")
response = query_engine.query("What is a decision tree?")
print(response)