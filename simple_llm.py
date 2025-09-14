from langchain_huggingface import HuggingFacePipeline , ChatHuggingFace
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import PromptTemplate
model_name = r"C:\Users\itsam\LangChain Models\TinyLlama-1.1B-Chat-v1.0"



llm = HuggingFacePipeline.from_model_id(
    model_id=model_name,     
    task="text-generation",
    model_kwargs={"temperature": 0.7, "max_length": 256}
)

model = ChatHuggingFace(llm=llm)


prompt = PromptTemplate(
    template='Suggest a catchy blog title abot {topic}.',
    input_variables=['topic']
)


topic = input('Enter a topic : ')

formatted_prompt = prompt.format(topic=topic)

blog_title = model.predict(formatted_prompt)

print('Generated Blog Title : ', blog_title)