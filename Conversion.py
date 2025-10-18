from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from langchain_core.tools import tool
import requests

# --- Define Tools ---

@tool
def get_conversion_factor(base_currency: str, target_currency: str) -> float:
    """
    Fetches the real-time currency conversion rate between a given base and target currency.
    """
    url = f'https://v6.exchangerate-api.com/v6/31536d9fc105b5a1bf602a3c/pair/{base_currency}/{target_currency}'
    response = requests.get(url).json()
    return response.get("conversion_rate", None)

@tool
def convert(base_currency_value: float, conversion_rate: float) -> float:
    """
    Converts the given base currency value to target currency using the conversion rate.
    """
    return base_currency_value * conversion_rate

# --- Load Local Model ---
model_name = r"C:\Users\itsam\LangChain Models\TinyLlama-1.1B-Chat-v1.0"

llm_pipeline = HuggingFacePipeline.from_model_id(
    model_id=model_name,
    task="text-generation",
    model_kwargs={"temperature": 0.7, "max_length": 256}
)
model = ChatHuggingFace(llm=llm_pipeline)

# --- Manual Tool Chaining ---
base_currency = "USD"
target_currency = "INR"
amount = 10

# Step 1: Get conversion rate
conversion_rate = get_conversion_factor.invoke({
    "base_currency": base_currency,
    "target_currency": target_currency
})

# Step 2: Convert amount using fetched rate
converted_amount = convert.invoke({
    "base_currency_value": amount,
    "conversion_rate": conversion_rate
})

# Step 3: Let the LLM format the answer
user_prompt = f"Convert {amount} {base_currency} to {target_currency} using the conversion rate {conversion_rate}."
response = model.invoke([{"type": "human", "content": user_prompt}])

# --- Show results ---
print(f"Conversion rate (USD → INR): {conversion_rate}")
print(f"{amount} {base_currency} = {converted_amount} {target_currency}")
print("\nLLM formatted response:\n", response.content)











# from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
# from langchain.agents import create_tool_calling_agent, AgentExecutor
# from langchain import hub
# from langchain_core.tools import tool
# import requests

# @tool
# def get_conversion_factor(base_currency: str, target_currency: str) -> float:
#     """
#     Fetches the currency conversion rate between a given base currency and a target currency.
#     """
#     url = f'https://v6.exchangerate-api.com/v6/31536d9fc105b5a1bf602a3c/pair/{base_currency}/{target_currency}'
#     response = requests.get(url)
#     data = response.json()
#     return data["conversion_rate"]

# @tool
# def convert(base_currency_value: int, conversion_rate: float) -> float:
#     """
#     Converts the given base currency value to target currency using the conversion rate.
#     """
#     return base_currency_value * conversion_rate

# # Model path (local)
# model_name = r"C:\Users\itsam\LangChain Models\TinyLlama-1.1B-Chat-v1.0"

# # Load the model pipeline
# llm = HuggingFacePipeline.from_model_id(
#     model_id=model_name,
#     task="text-generation",
#     model_kwargs={"temperature": 0.7, "max_length": 256}
# )

# model = ChatHuggingFace(llm=llm)

# # Define tools
# tools = [get_conversion_factor, convert]

# # Load a built-in prompt 
# prompt = hub.pull("hwchase17/openai-tools-agent")

# # Create tool-calling agent correctly
# agent = create_tool_calling_agent(model, tools, prompt)
# agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# # Run the pipeline
# response = agent_executor.invoke({
#     "input": "What is the conversion factor between USD and INR, and convert 10 USD to INR?"
# })

# print("\n Final Answer:\n", response)
