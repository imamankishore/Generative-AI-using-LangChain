from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

# Load your .env file
load_dotenv()

# Initialize DeepSeek LLM
llm = ChatOpenAI(
    model="deepseek-chat",  # or "deepseek-coder"
    openai_api_key=os.getenv("DEEPSEEK_API_KEY"),
    openai_api_base="https://api.deepseek.com"   # 👈 important
)

# Run a query
result = llm.invoke("What is the capital of India?")
print(result.content)
