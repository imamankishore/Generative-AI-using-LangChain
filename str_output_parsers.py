from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from langchain_core.prompts import PromptTemplate

model_name = r"C:\Users\itsam\LangChain Models\TinyLlama-1.1B-Chat-v1.0"


llm = HuggingFacePipeline.from_model_id(
    model_id=model_name,     
    task="text-generation",
    model_kwargs={"temperature": 0.7, "max_length": 256}
)

model = ChatHuggingFace(llm=llm)


#1st Prompt -->  detailed prompt

template1 = PromptTemplate(
    template='Write a detailed Report on {topic}',
    input_variables=['topic']
)


#2nd Prompt -->  summary prompt

template2 = PromptTemplate(
    template='WWrite a 5 line summary on text. /n {text}',
    input_variables=['text']
)


prompt1 = template1.invoke({'topic' : 'Black Hole'})

result = model.invoke(prompt1)


prompt2 = template2.invoke({'text' : result.content})

result1 = model.invoke(prompt2)


print(result1.content)