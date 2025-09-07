from langchain_huggingface import HuggingFacePipeline , ChatHuggingFace
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
model_name = r"C:\Users\itsam\LangChain Models\TinyLlama-1.1B-Chat-v1.0"



llm = HuggingFacePipeline.from_model_id(
    model_id=model_name,     
    task="text-generation",
    model_kwargs={"temperature": 0.7, "max_length": 256}
)

model = ChatHuggingFace(llm=llm)

chat_history = [
    SystemMessage(content='you are a helpful AI assistant')
]

while True:
    user_input = input("You: ")
    chat_history.append(HumanMessage(content=user_input))
    if user_input == 'exit':
        break
    result = model.invoke(chat_history)
    chat_history.append(AIMessage(content=result.content))
    print("AI: ", result.content)

print(chat_history)

