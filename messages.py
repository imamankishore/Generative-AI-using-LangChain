from langchain_core.messages import SystemMessage , HumanMessage, AIMessage
from langchain_huggingface import ChatHuggingFace , HuggingFacePipeline

model_name = r"C:\Users\itsam\LangChain Models\TinyLlama-1.1B-Chat-v1.0"



llm = HuggingFacePipeline.from_model_id(
    model_id=model_name,     
    task="text-generation",
    model_kwargs={"temperature": 0.7, "max_length": 256}
)

model = ChatHuggingFace(llm=llm)

messages = [
    SystemMessage(content='you are a Helpful assistant'),
    HumanMessage(content='Tell me about Langchain')
]

result = model.invoke(messages)

messages.append(AIMessage(content=result.content))

print(messages)